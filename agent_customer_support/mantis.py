"""MantisBT client: files a verified bug as an issue in the team's tracker.

Same shape as `escalation.Escalator` — httpx, settings-driven, constructor kwargs so
tests can inject values without touching the lru_cached Settings — and the same
opt-in rule: with no base URL or token the client logs and does nothing.

`create_issue` NEVER raises. By the time it runs the bug is verified and the reply is
about to be sent; a MantisBT outage must cost the ticket, not the turn. The backlog
row and the Zalo handoff still happen, and the Zalo note says the ticket is missing
so CS can file it by hand.

Files go in a second call on purpose: MantisBT rejects an upload over its
`max_file_size` as a whole-request error, and a ticket without its screenshot beats
no ticket at all.
"""

import logging

import httpx
from pydantic import BaseModel

from agent_customer_support.config import get_settings
from agent_customer_support.llm.schemas import BugReport
from agent_customer_support.observability import tracing

logger = logging.getLogger(__name__)

# MantisBT's `summary` column (the issue title) is VARCHAR(128).
_SUMMARY_MAX = 128


def format_title(customer_name: str, title: str) -> str:
    """`[BUG][<customer_name>] <title>`, fitted to MantisBT's 128-char summary.

    The team scans the issue list by customer, so the prefix is the part that must
    survive: when the whole thing is too long, only the title is cut.
    """
    prefix = f"[BUG][{customer_name}] "
    room = _SUMMARY_MAX - len(prefix)
    return f"{prefix}{title[: max(room, 0)]}"


class MantisFile(BaseModel):
    """One attachment, in memory only for the duration of the turn."""

    name: str
    content_b64: str


class MantisIssue(BaseModel):
    id: int
    url: str


class MantisClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_token: str | None = None,
        project: str | None = None,
        category: str | None = None,
        timeout_seconds: int | None = None,
        max_transcript_chars: int | None = None,
    ) -> None:
        cfg = get_settings()
        raw_base = base_url if base_url is not None else cfg.mantis_base_url
        self.base_url = raw_base.rstrip("/") if raw_base else None
        self.api_token = api_token if api_token is not None else cfg.mantis_api_token
        self.project = project if project is not None else cfg.mantis_project
        self.category = category if category is not None else cfg.mantis_category
        self.timeout = timeout_seconds or cfg.mantis_timeout_seconds
        self.max_transcript_chars = max_transcript_chars or cfg.mantis_max_transcript_chars

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.api_token)

    def issue_url(self, issue_id: int) -> str:
        return f"{self.base_url}/view.php?id={issue_id}"

    async def create_issue(
        self,
        *,
        report: BugReport,
        customer_id: str,
        customer_name: str,
        application: str | None,
        transcript: str,
        files: list[MantisFile],
    ) -> MantisIssue | None:
        if not self.enabled:
            logger.warning(
                "MantisBT not configured; bug ticket skipped for %s: %s", customer_id, report.title
            )
            return None
        with tracing.span(
            "tool.mantis.create_issue",
            as_type="tool",
            input={"title": report.title, "application": application, "files": len(files)},
        ) as sp:
            try:
                issue = await self._create(
                    report, customer_id, customer_name or customer_id, application, transcript
                )
            except Exception as exc:  # noqa: BLE001 - degrade, never break the turn
                logger.warning("MantisBT issue creation failed for %s: %s", customer_id, exc)
                sp.update(output={"error": str(exc)})
                return None
            attached = 0
            if files:
                try:
                    await self._attach(issue.id, files)
                    attached = len(files)
                except Exception as exc:  # noqa: BLE001 - the ticket exists; only the files are lost
                    logger.warning("MantisBT file upload failed for issue %s: %s", issue.id, exc)
            sp.update(output={"issue_id": issue.id, "url": issue.url, "files_attached": attached})
            return issue

    async def add_note(self, issue_id: int, text: str) -> bool:
        """Append a note to an existing issue -- how the user's contact details reach a
        ticket that was filed before they were given. Never raises."""
        if not self.enabled:
            return False
        with tracing.span(
            "tool.mantis.add_note", as_type="tool", input={"issue_id": issue_id}
        ) as sp:
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(
                        f"{self.base_url}/api/rest/issues/{issue_id}/notes",
                        json={"text": text},
                        headers=self._headers(),
                    )
                    resp.raise_for_status()
            except Exception as exc:  # noqa: BLE001 - degrade, never break the turn
                logger.warning("MantisBT note failed for issue %s: %s", issue_id, exc)
                sp.update(output={"error": str(exc)})
                return False
            sp.update(output={"ok": True})
            return True

    def _headers(self) -> dict[str, str]:
        # Never log these: the token mints issues as the bot user.
        return {"Authorization": self.api_token or "", "Content-Type": "application/json"}

    async def _create(
        self,
        report: BugReport,
        customer_id: str,
        customer_name: str,
        application: str | None,
        transcript: str,
    ) -> MantisIssue:
        header = "\n".join(
            [
                f"Khách hàng: {customer_name} ({customer_id})",
                f"Ứng dụng: {application or '(không rõ)'}",
                "Nguồn: chatbot hỗ trợ (tự động)",
            ]
        )
        body = {
            "summary": format_title(customer_name, report.title),
            "description": report.summary,
            "steps_to_reproduce": report.steps_to_reproduce,
            "additional_information": (
                f"{header}\n---\nHội thoại:\n{transcript[: self.max_transcript_chars]}"
            ),
            "project": {"name": self.project},
            "category": {"name": self.category},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/rest/issues", json=body, headers=self._headers()
            )
            resp.raise_for_status()
            issue_id = int(resp.json()["issue"]["id"])
        return MantisIssue(id=issue_id, url=self.issue_url(issue_id))

    async def _attach(self, issue_id: int, files: list[MantisFile]) -> None:
        body = {"files": [{"name": f.name, "content": f.content_b64} for f in files]}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/rest/issues/{issue_id}/files",
                json=body,
                headers=self._headers(),
            )
            resp.raise_for_status()

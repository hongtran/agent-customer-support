"""Email the CS team. Second notification channel next to the Zalo webhook.

Same shape as `escalation.Escalator` and `mantis.MantisClient`: httpx, settings-driven,
constructor kwargs so tests can inject values, off unless configured. Sent through
Resend's HTTP API (`POST /emails`) rather than its SDK, which is sync-only and would
put the call back on a worker thread.

`send` NEVER raises. By the time it runs the handoff has been posted to Zalo and the
reply is paid for; a mail provider problem must cost the email, not the turn.
"""

import logging

import httpx

from agent_customer_support.config import get_settings
from agent_customer_support.observability import tracing

logger = logging.getLogger(__name__)


class CSMailer:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_url: str | None = None,
        sender: str | None = None,
        to: str | None = None,
        cc: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        cfg = get_settings()
        self.api_key = api_key if api_key is not None else cfg.resend_api_key
        self.api_url = api_url if api_url is not None else cfg.resend_api_url
        self.sender = sender if sender is not None else cfg.cs_mail_from
        self.recipients = _split(to if to is not None else cfg.cs_mail_to)
        # Optional: CC alone does not enable the mailer, `to` is still required.
        self.cc = _split(cc if cc is not None else cfg.cs_mail_cc)
        self.timeout = timeout_seconds or cfg.cs_mail_timeout_seconds

    @property
    def enabled(self) -> bool:
        # Resend rejects a send with no `from`, so the sender is required too.
        return bool(self.api_key and self.sender and self.recipients)

    async def send(self, *, subject: str, body: str) -> bool:
        if not self.enabled:
            logger.info("CS mail not configured; skipped: %s", subject)
            return False
        # The body carries the transcript, so only the subject reaches the trace.
        with tracing.span(
            "tool.cs_mail.send",
            as_type="tool",
            input={
                "subject": subject,
                "recipients": len(self.recipients),
                "cc": len(self.cc),
            },
        ) as sp:
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(
                        self.api_url,
                        # Never log this header.
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json=self._payload(subject, body),
                    )
                if resp.is_error:
                    # Resend explains itself in `message` (e.g. an unverified `from`
                    # domain); the bare status code would not.
                    logger.warning(
                        "CS mail rejected (%s): %s %s",
                        subject,
                        resp.status_code,
                        _error_message(resp),
                    )
                    sp.update(output={"error": resp.status_code})
                    return False
            except Exception as exc:  # noqa: BLE001 - degrade, never break the turn
                logger.warning("CS mail failed (%s): %s", subject, exc)
                sp.update(output={"error": str(exc)})
                return False
            sp.update(output={"sent": True})
        return True

    def _payload(self, subject: str, body: str) -> dict[str, object]:
        payload: dict[str, object] = {
            "from": self.sender,
            "to": self.recipients,
            "subject": subject,
            "text": body,
        }
        # Left out when empty rather than sent as [], so a mailer with no CC sends
        # exactly the request it always did.
        if self.cc:
            payload["cc"] = self.cc
        return payload


def _split(raw: str) -> list[str]:
    """Comma-separated addresses → trimmed list, blanks dropped."""
    return [r.strip() for r in raw.split(",") if r.strip()]


def _error_message(resp: httpx.Response) -> str:
    try:
        return str(resp.json().get("message", ""))
    except Exception:  # noqa: BLE001 - a non-JSON error body
        return resp.text[:200]

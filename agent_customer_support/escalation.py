import logging

import httpx

from agent_customer_support.config import get_settings
from agent_customer_support.contact import describe
from agent_customer_support.mailer import CSMailer
from agent_customer_support.models import ContactInfo

logger = logging.getLogger(__name__)

# Zalo group messages are read on a phone; the email carries the full transcript.
_ZALO_TRANSCRIPT_MAX = 3000


class Escalator:
    """Fan-out to the CS team: the Zalo webhook and the CS mailbox.

    The Zalo post keeps its original contract -- an HTTP error still raises, so a
    broken webhook is loud in tests and logs. The email is best-effort on top of it:
    `CSMailer.send` never raises, and it runs even when no webhook is configured so a
    deployment with only a mailbox still gets every handoff.
    """

    def __init__(self, webhook_url: str | None = None, mailer: CSMailer | None = None) -> None:
        self.webhook_url = (
            webhook_url if webhook_url is not None else get_settings().zalo_cs_webhook_url
        )
        self.mailer = mailer if mailer is not None else CSMailer()

    async def escalate(
        self,
        *,
        customer_id: str,
        reason: str,
        transcript: str,
        note: str | None = None,
        cc: list[str] | None = None,
    ) -> None:
        """Post the handoff to CS.

        `note` is one extra line under the reason -- today the MantisBT ticket link for
        a verified bug, or the message that no ticket could be created so CS files it
        by hand. Kept generic so the transport knows nothing about tickets.
        """
        note_line = f"\n{note}" if note else ""
        head = f"[HỖ TRỢ] Khách {customer_id}\nLý do: {reason}{note_line}\n---\n"
        await self._post_zalo(head + transcript[:_ZALO_TRANSCRIPT_MAX], f"{customer_id}/{reason}")
        await self._mail(subject=f"[HỖ TRỢ][{customer_id}] {reason}", body=head + transcript, cc=cc)

    async def contact_update(
        self,
        *,
        customer_id: str,
        customer_name: str,
        reason: str | None,
        contact: ContactInfo,
        ticket_url: str | None,
    ) -> None:
        """Second notification, after the user answered the contact ask. Never raises:
        the handoff itself already went out; this only adds a phone number to it."""
        text = "\n".join(
            [
                f"[HỖ TRỢ] Khách {customer_name} ({customer_id}) đã để lại liên hệ",
                f"Lý do: {reason or '(không rõ)'}",
                f"Liên hệ: {describe(contact)}",
                f"Ticket: {ticket_url or '—'}",
                "---",
                contact.raw,
            ]
        )
        try:
            await self._post_zalo(text, f"{customer_id}/contact")
        except Exception as exc:  # noqa: BLE001 - degrade, never break the turn
            logger.warning("Zalo contact update failed for %s: %s", customer_id, exc)
        await self._mail(subject=f"[HỖ TRỢ][{customer_id}] Khách để lại liên hệ", body=text)

    async def _post_zalo(self, text: str, what: str) -> None:
        if not self.webhook_url:
            logger.warning("No Zalo CS webhook configured; escalation logged only: %s", what)
            return
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(self.webhook_url, json={"text": text})
            resp.raise_for_status()

    async def _mail(self, *, subject: str, body: str, cc: list[str] | None = None) -> None:
        if not self.mailer.enabled:
            return
        try:
            await self.mailer.send(subject=subject, body=body, cc=cc)
        except Exception as exc:  # noqa: BLE001 - the mailer should not raise; belt and braces
            logger.warning("CS mail failed (%s): %s", subject, exc)

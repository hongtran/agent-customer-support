"""Email the CS team. Second notification channel next to the Zalo webhook.

Same shape as `escalation.Escalator` and `mantis.MantisClient`: settings-driven,
constructor kwargs so tests can inject values, off unless configured. Stdlib
`smtplib` in a worker thread rather than a new async dependency -- one short
message per handoff does not justify one.

`send` NEVER raises. By the time it runs the handoff has been posted to Zalo and the
reply is paid for; a mail server problem must cost the email, not the turn.
"""

import asyncio
import logging
import smtplib
from email.message import EmailMessage

from agent_customer_support.config import get_settings

logger = logging.getLogger(__name__)


class CSMailer:
    def __init__(
        self,
        *,
        smtp_host: str | None = None,
        smtp_port: int | None = None,
        use_tls: bool | None = None,
        username: str | None = None,
        password: str | None = None,
        sender: str | None = None,
        to: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        cfg = get_settings()
        self.smtp_host = smtp_host if smtp_host is not None else cfg.cs_mail_smtp_host
        self.smtp_port = smtp_port if smtp_port is not None else cfg.cs_mail_smtp_port
        self.use_tls = use_tls if use_tls is not None else cfg.cs_mail_use_tls
        self.username = username if username is not None else cfg.cs_mail_username
        self.password = password if password is not None else cfg.cs_mail_password
        self.sender = sender if sender is not None else cfg.cs_mail_from
        raw_to = to if to is not None else cfg.cs_mail_to
        self.recipients = [r.strip() for r in raw_to.split(",") if r.strip()]
        self.timeout = timeout_seconds or cfg.cs_mail_timeout_seconds

    @property
    def enabled(self) -> bool:
        return bool(self.smtp_host and self.recipients)

    async def send(self, *, subject: str, body: str) -> bool:
        if not self.enabled:
            logger.info("CS mail not configured; skipped: %s", subject)
            return False
        try:
            await asyncio.to_thread(self._send_sync, subject, body)
        except Exception as exc:  # noqa: BLE001 - degrade, never break the turn
            logger.warning("CS mail failed (%s): %s", subject, exc)
            return False
        return True

    def _send_sync(self, subject: str, body: str) -> None:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self.sender or (self.username or "")
        msg["To"] = ", ".join(self.recipients)
        msg.set_content(body)
        with smtplib.SMTP(self.smtp_host or "", self.smtp_port, timeout=self.timeout) as smtp:
            if self.use_tls:
                smtp.starttls()
            if self.username:
                # Never log these.
                smtp.login(self.username, self.password or "")
            smtp.send_message(msg)

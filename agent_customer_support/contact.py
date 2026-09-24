"""Pull a phone number and an email out of a free-text reply.

Deliberately regex, not an LLM call: the reply to "để lại SĐT và email" is short and
the two shapes are unambiguous, so a model would add cost and a None case for no
gain. `parse` never raises and never returns None -- a miss is a `ContactInfo` with
both fields None and `raw` kept, so CS still sees what the user wrote.
"""

import re

from agent_customer_support.models import ContactInfo

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# Applied AFTER separators are stripped: `0912 345 678`, `0912.345.678`,
# `(024) 3825-1234` all collapse to one digit run. VN mobiles are 10 digits, landlines
# 11, both starting with 0; the international form is +84 / 84 followed by the rest.
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?84|0)(\d{9,10})(?!\d)")
_SEPARATORS_RE = re.compile(r"[\s.\-()]")


def parse(text: str) -> ContactInfo:
    email_match = _EMAIL_RE.search(text)
    email = email_match.group(0).lower() if email_match else None
    # Remove the email before looking for digits so `user0912345678@mail.com` is an
    # email, not a phone number.
    rest = text[: email_match.start()] + text[email_match.end() :] if email_match else text
    phone_match = _PHONE_RE.search(_SEPARATORS_RE.sub("", rest))
    phone = f"0{phone_match.group(1)}" if phone_match else None
    return ContactInfo(phone=phone, email=email, raw=text)


def describe(contact: ContactInfo) -> str:
    """One line for a Zalo message, an email or a ticket note."""
    parts = []
    if contact.phone:
        parts.append(f"SĐT: {contact.phone}")
    if contact.email:
        parts.append(f"Email: {contact.email}")
    return " · ".join(parts) if parts else "chưa cung cấp"

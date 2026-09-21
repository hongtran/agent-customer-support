from unittest.mock import patch

import pytest

from agent_customer_support.mailer import CSMailer

pytestmark = pytest.mark.asyncio


class _FakeSMTP:
    """Records what the mailer does; stands in for smtplib.SMTP."""

    instances: list["_FakeSMTP"] = []

    def __init__(self, host, port, timeout=None):
        self.host, self.port, self.timeout = host, port, timeout
        self.calls: list[tuple] = []
        self.sent = []
        _FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.calls.append(("quit",))
        return False

    def starttls(self):
        self.calls.append(("starttls",))

    def login(self, user, password):
        self.calls.append(("login", user, password))

    def send_message(self, msg):
        self.sent.append(msg)


def _mailer(**kw) -> CSMailer:
    return CSMailer(
        smtp_host=kw.pop("smtp_host", "smtp.example"),
        smtp_port=587,
        use_tls=kw.pop("use_tls", True),
        username=kw.pop("username", "bot"),
        password=kw.pop("password", "pw"),
        sender=kw.pop("sender", "bot@example.vn"),
        to=kw.pop("to", "cs1@example.vn, cs2@example.vn"),
        **kw,
    )


@pytest.fixture(autouse=True)
def _reset():
    _FakeSMTP.instances.clear()


async def test_send_delivers_to_every_recipient_with_subject_and_body():
    with patch("agent_customer_support.mailer.smtplib.SMTP", _FakeSMTP):
        ok = await _mailer().send(subject="[HỖ TRỢ] test", body="nội dung\nhai dòng")
    assert ok is True
    smtp = _FakeSMTP.instances[0]
    assert (smtp.host, smtp.port) == ("smtp.example", 587)
    assert ("starttls",) in smtp.calls
    assert ("login", "bot", "pw") in smtp.calls
    msg = smtp.sent[0]
    assert msg["Subject"] == "[HỖ TRỢ] test"
    assert msg["From"] == "bot@example.vn"
    assert msg["To"] == "cs1@example.vn, cs2@example.vn"
    assert "hai dòng" in msg.get_content()


async def test_no_tls_and_no_login_when_not_configured():
    with patch("agent_customer_support.mailer.smtplib.SMTP", _FakeSMTP):
        await _mailer(use_tls=False, username="", password="").send(subject="s", body="b")
    calls = _FakeSMTP.instances[0].calls
    assert ("starttls",) not in calls
    assert not any(c[0] == "login" for c in calls)


async def test_smtp_failure_returns_false_without_raising():
    class _Boom(_FakeSMTP):
        def send_message(self, msg):
            raise OSError("smtp down")

    with patch("agent_customer_support.mailer.smtplib.SMTP", _Boom):
        assert await _mailer().send(subject="s", body="b") is False


async def test_disabled_without_host_or_recipients_makes_no_connection():
    with patch("agent_customer_support.mailer.smtplib.SMTP", _FakeSMTP):
        m = _mailer(smtp_host="")
        assert m.enabled is False
        assert await m.send(subject="s", body="b") is False
        m2 = _mailer(to="")
        assert m2.enabled is False
        assert await m2.send(subject="s", body="b") is False
    assert _FakeSMTP.instances == []


async def test_recipients_list_is_split_and_trimmed():
    m = _mailer(to=" a@x.vn ,b@x.vn,, ")
    assert m.recipients == ["a@x.vn", "b@x.vn"]

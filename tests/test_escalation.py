import pytest
import respx
import httpx
from agent_customer_support.escalation import Escalator

pytestmark = pytest.mark.asyncio


@respx.mock
async def test_escalate_posts_to_zalo_webhook():
    route = respx.post("https://zalo.example/cs").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    esc = Escalator(webhook_url="https://zalo.example/cs")
    await esc.escalate(customer_id="c1", reason="bế tắc", transcript="u: hi")
    assert route.called
    sent = route.calls[0].request.content.decode()
    assert "c1" in sent and "bế tắc" in sent


async def test_escalate_noop_when_no_webhook():
    esc = Escalator(webhook_url=None)
    await esc.escalate(customer_id="c1", reason="x", transcript="")  # must not raise


@respx.mock
async def test_escalate_includes_note_line_when_given():
    route = respx.post("https://zalo.example/cs").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    esc = Escalator(webhook_url="https://zalo.example/cs")
    await esc.escalate(
        customer_id="c1",
        reason="verified bug",
        transcript="u: hi",
        note="Ticket MantisBT: https://mantis.example/view.php?id=7",
    )
    sent = route.calls[0].request.content.decode()
    assert "https://mantis.example/view.php?id=7" in sent


@respx.mock
async def test_escalate_without_note_has_no_ticket_line():
    route = respx.post("https://zalo.example/cs").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    esc = Escalator(webhook_url="https://zalo.example/cs")
    await esc.escalate(customer_id="c1", reason="x", transcript="u: hi")
    assert "Ticket" not in route.calls[0].request.content.decode()


@respx.mock
async def test_escalate_also_emails_cs_with_full_transcript():
    from unittest.mock import AsyncMock

    respx.post("https://zalo.example/cs").mock(return_value=httpx.Response(200, json={}))
    mailer = AsyncMock()
    mailer.enabled = True
    esc = Escalator(webhook_url="https://zalo.example/cs", mailer=mailer)
    long_transcript = "user: " + "x" * 5000
    await esc.escalate(
        customer_id="c1", reason="verified bug", transcript=long_transcript, note="Ticket: t"
    )
    mailer.send.assert_awaited_once()
    kw = mailer.send.call_args.kwargs
    assert kw["subject"] == "[HỖ TRỢ][c1] verified bug"
    assert "Ticket: t" in kw["body"]
    assert "x" * 5000 in kw["body"]  # email is not capped like Zalo


async def test_escalate_emails_even_without_zalo_webhook():
    from unittest.mock import AsyncMock

    mailer = AsyncMock()
    mailer.enabled = True
    esc = Escalator(webhook_url=None, mailer=mailer)
    await esc.escalate(customer_id="c1", reason="x", transcript="u: hi")
    mailer.send.assert_awaited_once()


@respx.mock
async def test_mailer_failure_does_not_fail_escalate():
    from unittest.mock import AsyncMock

    route = respx.post("https://zalo.example/cs").mock(return_value=httpx.Response(200, json={}))
    mailer = AsyncMock()
    mailer.enabled = True
    mailer.send.side_effect = RuntimeError("boom")
    esc = Escalator(webhook_url="https://zalo.example/cs", mailer=mailer)
    await esc.escalate(customer_id="c1", reason="x", transcript="u: hi")  # must not raise
    assert route.called


@respx.mock
async def test_contact_update_notifies_both_channels():
    from unittest.mock import AsyncMock

    from agent_customer_support.models import ContactInfo

    route = respx.post("https://zalo.example/cs").mock(return_value=httpx.Response(200, json={}))
    mailer = AsyncMock()
    mailer.enabled = True
    esc = Escalator(webhook_url="https://zalo.example/cs", mailer=mailer)
    await esc.contact_update(
        customer_id="c1",
        customer_name="Công ty ABC",
        reason="verified bug",
        contact=ContactInfo(phone="0912345678", email="a@b.vn", raw="0912345678 a@b.vn gọi sau 5h"),
        ticket_url="https://mantis.example/view.php?id=7",
    )
    zalo = route.calls[0].request.content.decode()
    for needle in (
        "Công ty ABC",
        "c1",
        "verified bug",
        "0912345678",
        "a@b.vn",
        "https://mantis.example/view.php?id=7",
        "gọi sau 5h",
    ):
        assert needle in zalo
    kw = mailer.send.call_args.kwargs
    assert "0912345678" in kw["body"] and "a@b.vn" in kw["body"]
    assert kw["subject"] == "[HỖ TRỢ][c1] Khách để lại liên hệ"


@respx.mock
async def test_contact_update_never_raises():
    from unittest.mock import AsyncMock

    from agent_customer_support.models import ContactInfo

    respx.post("https://zalo.example/cs").mock(side_effect=httpx.ConnectError("down"))
    mailer = AsyncMock()
    mailer.enabled = True
    esc = Escalator(webhook_url="https://zalo.example/cs", mailer=mailer)
    await esc.contact_update(
        customer_id="c1",
        customer_name="X",
        reason=None,
        contact=ContactInfo(phone="0912345678"),
        ticket_url=None,
    )  # must not raise
    mailer.send.assert_awaited_once()

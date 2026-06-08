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

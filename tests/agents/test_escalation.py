import pytest
from unittest.mock import AsyncMock
from agent_customer_support.agents.escalation import EscalationAgent
from agent_customer_support.agents.context import TurnContext
from agent_customer_support.models import CustomerProfile, SessionState, Conversation

pytestmark = pytest.mark.asyncio


def _ctx(escalator, manager_email: str | None = None) -> TurnContext:
    return TurnContext(
        customer=CustomerProfile(customer_id="c1", name="C1", manager_email=manager_email),
        session=SessionState(conversation_id="cv1"),
        conversation=Conversation(conversation_id="cv1", customer_id="c1"),
        message="cho gặp nhân viên",
        transcript="user: cho gặp nhân viên",
        escalator=escalator,
    )


async def test_escalation_calls_escalator_and_returns_escalated():
    escalator = AsyncMock()
    agent = EscalationAgent()
    res = await agent.run(_ctx(escalator), reason="knowledge unresolved")
    assert res.escalated is True
    assert res.reply
    escalator.escalate.assert_awaited_once()
    kwargs = escalator.escalate.call_args.kwargs
    assert kwargs["customer_id"] == "c1"
    assert kwargs["reason"] == "knowledge unresolved"


async def test_escalation_forwards_note_to_escalator():
    escalator = AsyncMock()
    await EscalationAgent().run(_ctx(escalator), reason="verified bug", note="Ticket: x")
    assert escalator.escalate.call_args.kwargs["note"] == "Ticket: x"


async def test_escalation_without_note_passes_none():
    escalator = AsyncMock()
    await EscalationAgent().run(_ctx(escalator), reason="clarify limit")
    assert escalator.escalate.call_args.kwargs["note"] is None


async def test_escalation_result_names_the_reason():
    res = await EscalationAgent().run(_ctx(AsyncMock()), reason="knowledge unresolved")
    assert res.escalation_reason == "knowledge unresolved"


@pytest.mark.parametrize(
    "reason", ["user requested human", "hop limit", "bug loop", "ungrounded answer"]
)
async def test_other_reasons_reply_without_notifying_cs(reason):
    escalator = AsyncMock()
    res = await EscalationAgent().run(_ctx(escalator, "lead@cenlab.vn"), reason=reason)
    escalator.escalate.assert_not_called()
    # Same shape as a notified handoff, so the contact gate and tracing still work.
    assert res.escalated is True and res.reply and res.escalation_reason == reason


@pytest.mark.parametrize("reason", ["verified bug", "knowledge unresolved", "clarify limit"])
async def test_manager_is_cced_on_notified_reasons(reason):
    escalator = AsyncMock()
    await EscalationAgent().run(_ctx(escalator, "lead@cenlab.vn"), reason=reason)
    assert escalator.escalate.call_args.kwargs["cc"] == ["lead@cenlab.vn"]


async def test_no_manager_means_no_cc():
    escalator = AsyncMock()
    await EscalationAgent().run(_ctx(escalator), reason="verified bug")
    assert escalator.escalate.call_args.kwargs["cc"] is None

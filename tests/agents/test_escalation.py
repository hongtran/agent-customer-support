import pytest
from unittest.mock import AsyncMock
from agent_customer_support.agents.escalation import EscalationAgent
from agent_customer_support.agents.context import TurnContext
from agent_customer_support.models import CustomerProfile, SessionState, Conversation

pytestmark = pytest.mark.asyncio


def _ctx(escalator) -> TurnContext:
    return TurnContext(
        customer=CustomerProfile(customer_id="c1", name="C1"),
        session=SessionState(conversation_id="cv1"),
        conversation=Conversation(conversation_id="cv1", customer_id="c1"),
        message="cho gặp nhân viên",
        transcript="user: cho gặp nhân viên",
        escalator=escalator,
    )


async def test_escalation_calls_escalator_and_returns_escalated():
    escalator = AsyncMock()
    agent = EscalationAgent()
    res = await agent.run(_ctx(escalator), reason="user asked")
    assert res.escalated is True
    assert res.reply
    escalator.escalate.assert_awaited_once()
    kwargs = escalator.escalate.call_args.kwargs
    assert kwargs["customer_id"] == "c1"
    assert kwargs["reason"] == "user asked"


async def test_escalation_forwards_note_to_escalator():
    escalator = AsyncMock()
    await EscalationAgent().run(_ctx(escalator), reason="verified bug", note="Ticket: x")
    assert escalator.escalate.call_args.kwargs["note"] == "Ticket: x"


async def test_escalation_without_note_passes_none():
    escalator = AsyncMock()
    await EscalationAgent().run(_ctx(escalator), reason="user asked")
    assert escalator.escalate.call_args.kwargs["note"] is None


async def test_escalation_result_names_the_reason():
    res = await EscalationAgent().run(_ctx(AsyncMock()), reason="knowledge unresolved")
    assert res.escalation_reason == "knowledge unresolved"

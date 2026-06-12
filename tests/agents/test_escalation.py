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

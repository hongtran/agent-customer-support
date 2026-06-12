from unittest.mock import AsyncMock
from agent_customer_support.agents.context import TurnContext
from agent_customer_support.agents.base import Agent
from agent_customer_support.models import (
    AgentResult, CustomerProfile, SessionState, Conversation,
)


def _ctx() -> TurnContext:
    return TurnContext(
        customer=CustomerProfile(customer_id="c1", name="C1", enabled_modules=["m"]),
        session=SessionState(conversation_id="cv1"),
        conversation=Conversation(conversation_id="cv1", customer_id="c1"),
        message="hi",
        attachments=[],
        transcript="user: hi",
        rag=AsyncMock(),
        flow_store=AsyncMock(),
        backlog=AsyncMock(),
        escalator=AsyncMock(),
    )


def test_turn_context_fields():
    ctx = _ctx()
    assert ctx.message == "hi"
    assert ctx.customer.customer_id == "c1"
    assert ctx.session.conversation_id == "cv1"


def test_agent_protocol_is_runtime_checkable():
    class Dummy:
        name = "dummy"

        async def run(self, ctx: TurnContext) -> AgentResult:
            return AgentResult(reply="ok")

    assert isinstance(Dummy(), Agent)

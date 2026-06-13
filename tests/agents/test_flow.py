import pytest
from unittest.mock import patch, AsyncMock
from agent_customer_support.agents.flow import FlowAgent, parse_goto
from agent_customer_support.agents.context import TurnContext
from agent_customer_support.models import (
    CustomerProfile, SessionState, Conversation, Flow, FlowStep, FlowTransition,
    FlowOutcome,
)

pytestmark = pytest.mark.asyncio


def _flow():
    return Flow(id="f1", title="T", module="m", steps=[
        FlowStep(id="s1", say="bước 1",
                 next=[FlowTransition(when="xong", goto="s2")]),
        FlowStep(id="s2", say="bước 2",
                 next=[FlowTransition(when="ok", goto="done")]),
    ], outcomes={"done": FlowOutcome(type="success", say="Hoàn tất")})


def _ctx(session, flow_store) -> TurnContext:
    return TurnContext(
        customer=CustomerProfile(customer_id="c1", name="C1", enabled_modules=["m"]),
        session=session,
        conversation=Conversation(conversation_id="cv1", customer_id="c1"),
        message="xong rồi", transcript="user: xong rồi",
        rag=AsyncMock(), flow_store=flow_store, backlog=AsyncMock(),
        escalator=AsyncMock(),
    )


def test_parse_goto():
    assert parse_goto("tiếp tục [[goto:s2]]") == ("tiếp tục", "s2")


async def test_advances_active_flow_step():
    fs = AsyncMock()
    fs.get.return_value = _flow()
    session = SessionState(conversation_id="cv1", active_flow_id="f1",
                           current_step_id="s1")
    seq = [{"stop_reason": "end_turn", "text": "Làm bước 2 nhé [[goto:s2]]",
            "tool_calls": []}]
    with patch("agent_customer_support.agents.flow.complete_with_tools",
               side_effect=seq):
        res = await FlowAgent().run(_ctx(session, fs))
    assert res.new_session.current_step_id == "s2"
    assert "[[goto" not in res.reply


async def test_outcome_clears_session():
    fs = AsyncMock()
    fs.get.return_value = _flow()
    session = SessionState(conversation_id="cv1", active_flow_id="f1",
                           current_step_id="s2")
    seq = [{"stop_reason": "end_turn", "text": "Tốt [[goto:done]]", "tool_calls": []}]
    with patch("agent_customer_support.agents.flow.complete_with_tools",
               side_effect=seq):
        res = await FlowAgent().run(_ctx(session, fs))
    assert res.new_session.active_flow_id is None
    assert res.new_session.current_step_id is None

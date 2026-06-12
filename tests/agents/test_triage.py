import pytest
from unittest.mock import patch, AsyncMock
from agent_customer_support.agents.triage import TriageAgent
from agent_customer_support.agents.context import TurnContext
from agent_customer_support.models import CustomerProfile, SessionState, Conversation

pytestmark = pytest.mark.asyncio


def _ctx(message, session=None) -> TurnContext:
    return TurnContext(
        customer=CustomerProfile(customer_id="c1", name="C1"),
        session=session or SessionState(conversation_id="cv1"),
        conversation=Conversation(conversation_id="cv1", customer_id="c1"),
        message=message,
        transcript=f"user: {message}",
        rag=AsyncMock(), flow_store=AsyncMock(),
    )


async def test_active_flow_routes_to_flow_without_llm():
    s = SessionState(conversation_id="cv1", active_flow_id="f1", current_step_id="s1")
    res = await TriageAgent().run(_ctx("ok rồi", s))
    assert res.action == "route" and res.routed_to == "flow"


async def test_explicit_human_request_routes_escalate():
    res = await TriageAgent().run(_ctx("cho tôi gặp nhân viên"))
    assert res.action == "route" and res.routed_to == "escalate"


async def test_ambiguous_message_clarifies():
    with patch("agent_customer_support.agents.triage.complete_text",
               return_value='{"action":"clarify","question":"Bạn muốn làm gì cụ thể?"}'):
        res = await TriageAgent().run(_ctx("phần mềm có vấn đề"))
    assert res.action == "reply"
    assert "cụ thể" in res.reply


async def test_clear_intent_routes_knowledge():
    with patch("agent_customer_support.agents.triage.complete_text",
               return_value='{"action":"route","target":"knowledge"}'):
        res = await TriageAgent().run(_ctx("cách tạo phiếu yêu cầu thử nghiệm?"))
    assert res.action == "route" and res.routed_to == "knowledge"

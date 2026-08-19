import pytest
from unittest.mock import patch, AsyncMock
from agent_customer_support.agents.triage import TriageAgent
from agent_customer_support.llm.schemas import TriageDecision
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
        rag=AsyncMock(),
        flow_store=AsyncMock(),
    )


async def test_active_flow_routes_to_flow_without_llm():
    s = SessionState(conversation_id="cv1", active_flow_id="f1", current_step_id="s1")
    res = await TriageAgent().run(_ctx("ok rồi", s))
    assert res.action == "route" and res.routed_to == "flow"


async def test_explicit_human_request_routes_escalate():
    res = await TriageAgent().run(_ctx("cho tôi gặp nhân viên"))
    assert res.action == "route" and res.routed_to == "escalate"


async def test_ambiguous_message_still_routes_to_knowledge():
    """Triage no longer clarifies — a vague message routes to knowledge, which owns
    clarification now that it has the RAG (and screenshot) context to do it well."""
    with patch(
        "agent_customer_support.agents.triage.complete_structured",
        return_value=TriageDecision(target="knowledge"),
    ):
        res = await TriageAgent().run(_ctx("phần mềm có vấn đề"))
    assert res.action == "route" and res.routed_to == "knowledge"


async def test_clear_intent_routes_knowledge():
    with patch(
        "agent_customer_support.agents.triage.complete_structured",
        return_value=TriageDecision(target="knowledge"),
    ):
        res = await TriageAgent().run(_ctx("cách tạo phiếu yêu cầu thử nghiệm?"))
    assert res.action == "route" and res.routed_to == "knowledge"


async def test_triage_passes_full_history_on_followup():
    """Triage sends the full conversation (not just ctx.message) so it can route a
    terse follow-up using the context of the turns before it."""
    from agent_customer_support.models import Turn

    ctx = _ctx("màn hình trắng")
    ctx.conversation.turns = [
        Turn(role="user", content="module thanh toán bị lỗi"),
        Turn(role="assistant", content="Lỗi gì vậy bạn?"),
    ]
    captured: dict = {}

    def fake_complete(*, messages, schema, system=None, model=None):
        captured["messages"] = messages
        captured["schema"] = schema
        return TriageDecision(target="knowledge")

    with patch(
        "agent_customer_support.agents.triage.complete_structured", side_effect=fake_complete
    ):
        await TriageAgent().run(ctx)

    # should have 3 messages: 2 prior turns + current
    assert len(captured["messages"]) == 3
    assert captured["messages"][0]["role"] == "user"
    assert captured["messages"][1]["role"] == "assistant"
    assert captured["messages"][2]["content"] == "màn hình trắng"
    assert captured["schema"] is TriageDecision


async def test_no_decision_falls_back_to_knowledge():
    """complete_structured returns None on an API error, refusal or truncation.
    Knowledge is the fail-safe: cheapest route, and the most reversible."""
    with patch("agent_customer_support.agents.triage.complete_structured", return_value=None):
        res = await TriageAgent().run(_ctx("phần mềm có vấn đề"))
    assert res.action == "route" and res.routed_to == "knowledge"


async def test_escalate_decision_is_honoured():
    with patch(
        "agent_customer_support.agents.triage.complete_structured",
        return_value=TriageDecision(target="escalate"),
    ):
        res = await TriageAgent().run(_ctx("việc này quá phức tạp"))
    assert res.action == "route" and res.routed_to == "escalate"


async def test_off_topic_routes_out_of_scope():
    with patch(
        "agent_customer_support.agents.triage.complete_structured",
        return_value=TriageDecision(target="out_of_scope"),
    ):
        res = await TriageAgent().run(_ctx("Gợi ý giúp tôi vài quán ăn trưa ngon gần văn phòng."))
    assert res.action == "route" and res.routed_to == "out_of_scope"

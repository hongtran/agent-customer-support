import pytest
from unittest.mock import AsyncMock, MagicMock
from agent_customer_support.agents.coordinator import Coordinator
from agent_customer_support.models import (
    AgentResult,
    CustomerProfile,
    SessionState,
    Conversation,
)

pytestmark = pytest.mark.asyncio


def _coord():
    c = Coordinator()
    c.customers = AsyncMock()
    c.customers.get.return_value = CustomerProfile(
        customer_id="c1", name="C1", enabled_modules=["m"]
    )
    c.conversations = AsyncMock()
    c.conversations.load.return_value = Conversation(conversation_id="cv1", customer_id="c1")
    c.sessions = AsyncMock()
    c.sessions.get.return_value = SessionState(conversation_id="cv1")
    c.rag = AsyncMock()
    c.flow_store = AsyncMock()
    c.backlog = AsyncMock()
    c.escalator = AsyncMock()
    # agent stubs
    c.guardrail = MagicMock()
    c.guardrail.check_input = AsyncMock(return_value={"pass": True, "reason": ""})
    c.guardrail.check_output = AsyncMock(return_value={"pass": True, "reason": ""})
    c.triage = MagicMock()
    c.knowledge = MagicMock()
    c.flow = MagicMock()
    c.verification = MagicMock()
    c.escalation = MagicMock()
    return c


async def test_input_guardrail_block_short_circuits():
    c = _coord()
    c.guardrail.check_input = AsyncMock(return_value={"pass": False, "reason": "empty"})
    c.triage.run = AsyncMock()
    res = await c.handle_turn(customer_id="c1", conversation_id="cv1", message="  ", attachments=[])
    assert res.escalated is False
    c.triage.run.assert_not_called()


async def test_knowledge_clarify_resume_bypasses_triage():
    """When a knowledge clarification is pending, the next turn goes straight to
    knowledge (the answer to our clarify question), not back through triage."""
    c = _coord()
    c.sessions.get.return_value = SessionState(conversation_id="cv1", pending="knowledge_clarify")
    c.triage.run = AsyncMock()
    c.knowledge.run = AsyncMock(return_value=AgentResult(reply="Vào menu X.", resolved=True))
    res = await c.handle_turn(
        customer_id="c1", conversation_id="cv1", message="trang xuất kho", attachments=[]
    )
    assert res.reply == "Vào menu X."
    c.triage.run.assert_not_called()
    c.knowledge.run.assert_awaited_once()


async def test_knowledge_clarify_roundtrip_no_escalate_then_resolves():
    """End-to-end clarify round-trip through the real KnowledgeAgent:

    Turn 1: compose returns a [[clarify]] marker → resolved=None. The turn must NOT
            escalate and must persist session.pending == "knowledge_clarify".
    Turn 2: compose returns a plain grounded answer → the reply is the answer and
            the pending flag is cleared.
    """
    from agent_customer_support.agents.knowledge import KnowledgeAgent
    from agent_customer_support.agents import knowledge as knowledge_mod

    c = _coord()
    # Use the real KnowledgeAgent so the clarify wiring (pending flag, resolved=None)
    # is exercised, not stubbed away.
    c.knowledge = KnowledgeAgent()
    c.triage.run = AsyncMock(return_value=AgentResult(action="route", routed_to="knowledge"))
    c.rag.search = AsyncMock(return_value={"passages": [], "citations": []})
    c.rag.search_with_fallback = c.rag.search  # product search entry point

    # A single mutable session that survives across both turns, mirroring how the
    # real SessionStore would carry pending state turn-to-turn.
    session = SessionState(conversation_id="cv1")
    c.sessions.get.return_value = session

    # Turn 1: compose emits a clarify marker.
    compose_outputs = iter(
        [
            "Bạn muốn tạo loại phiếu nào? [[clarify]]",
            "Vào menu Phiếu > Tạo mới.",
        ]
    )

    def fake_complete_text(*args, **kwargs):
        return next(compose_outputs)

    original_complete_text = knowledge_mod.complete_text
    knowledge_mod.complete_text = fake_complete_text
    try:
        res1 = await c.handle_turn(
            customer_id="c1", conversation_id="cv1", message="tạo phiếu", attachments=[]
        )
        assert res1.escalated is False
        assert "Bạn muốn tạo loại phiếu nào?" in res1.reply
        saved1 = c.sessions.save.call_args.args[0]
        assert saved1.pending == "knowledge_clarify"

        # Turn 2: same session, pending flag set → coordinator resumes in knowledge.
        c.sessions.get.return_value = saved1
        res2 = await c.handle_turn(
            customer_id="c1", conversation_id="cv1", message="phiếu kết quả", attachments=[]
        )
        assert res2.escalated is False
        assert res2.reply == "Vào menu Phiếu > Tạo mới."
        saved2 = c.sessions.save.call_args.args[0]
        assert saved2.pending is None
    finally:
        knowledge_mod.complete_text = original_complete_text


async def test_root_trace_uses_conversation_as_session(monkeypatch):
    from contextlib import contextmanager
    from agent_customer_support.agents import coordinator as coord_mod

    captured = {}

    @contextmanager
    def fake_trace(name, *, session_id=None, user_id=None, tags=None, input=None, metadata=None):
        captured["session_id"] = session_id
        captured["user_id"] = user_id
        yield MagicMock()

    monkeypatch.setattr(coord_mod.tracing, "trace", fake_trace)
    c = _coord()
    c.triage.run = AsyncMock(return_value=AgentResult(action="route", routed_to="knowledge"))
    c.knowledge.run = AsyncMock(return_value=AgentResult(reply="ok", resolved=True))
    await c.handle_turn(customer_id="c1", conversation_id="cv1", message="?", attachments=[])
    assert captured["session_id"] == "cv1"
    assert captured["user_id"] == "c1"


async def test_knowledge_resolved_returns_reply():
    c = _coord()
    c.triage.run = AsyncMock(return_value=AgentResult(action="route", routed_to="knowledge"))
    c.knowledge.run = AsyncMock(return_value=AgentResult(reply="đáp án", resolved=True))
    res = await c.handle_turn(
        customer_id="c1", conversation_id="cv1", message="cách làm X", attachments=[]
    )
    assert res.reply == "đáp án"


async def test_knowledge_unresolved_escalates():
    c = _coord()
    c.triage.run = AsyncMock(return_value=AgentResult(action="route", routed_to="knowledge"))
    c.knowledge.run = AsyncMock(return_value=AgentResult(reply="", resolved=False))
    c.escalation.run = AsyncMock(return_value=AgentResult(reply="chuyển nhân viên", escalated=True))
    res = await c.handle_turn(
        customer_id="c1", conversation_id="cv1", message="lỗi lạ", attachments=[]
    )
    assert res.escalated is True
    c.escalation.run.assert_awaited_once()


async def test_suspected_bug_starts_verification():
    c = _coord()
    c.triage.run = AsyncMock(return_value=AgentResult(action="route", routed_to="knowledge"))
    c.knowledge.run = AsyncMock(
        return_value=AgentResult(
            reply="nghi lỗi",
            resolved=False,
            suspected_bug=True,
            evidence={"module": "m", "summary": "A lỗi"},
        )
    )
    c.verification.run = AsyncMock(
        return_value=AgentResult(reply="gửi ảnh giúp mình", evidence_complete=False)
    )
    res = await c.handle_turn(
        customer_id="c1", conversation_id="cv1", message="A bị lỗi", attachments=[]
    )
    assert "ảnh" in res.reply
    saved = c.sessions.save.call_args.args[0]
    assert saved.pending == "verify_issue"


async def test_pending_verification_resumes_and_escalates_when_complete():
    c = _coord()
    c.sessions.get.return_value = SessionState(
        conversation_id="cv1",
        pending="verify_issue",
        pending_context={"module": "m", "summary": "A lỗi"},
    )
    c.verification.run = AsyncMock(
        return_value=AgentResult(
            reply="đã đủ",
            evidence_complete=True,
            evidence={"module": "m", "summary": "A lỗi", "has_image": True},
        )
    )
    c.escalation.run = AsyncMock(return_value=AgentResult(reply="chuyển nhân viên", escalated=True))
    res = await c.handle_turn(
        customer_id="c1", conversation_id="cv1", message="đây là ảnh", attachments=[]
    )
    assert res.escalated is True
    c.backlog.add.assert_awaited_once()
    saved = c.sessions.save.call_args.args[0]
    assert saved.pending is None


async def test_out_of_scope_route_refuses_before_knowledge():
    """Triage flags off-topic: canonical refusal, no RAG/knowledge spend, no escalation."""
    from agent_customer_support.agents.prompts import OUT_OF_SCOPE_REPLY

    c = _coord()
    c.triage.run = AsyncMock(return_value=AgentResult(action="route", routed_to="out_of_scope"))
    c.knowledge.run = AsyncMock()
    c.escalation.run = AsyncMock()
    res = await c.handle_turn(
        customer_id="c1", conversation_id="cv1", message="Tỷ giá USD hôm nay?", attachments=[]
    )
    assert res.reply == OUT_OF_SCOPE_REPLY
    assert res.escalated is False
    c.knowledge.run.assert_not_called()
    c.escalation.run.assert_not_called()

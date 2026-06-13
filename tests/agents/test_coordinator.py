import pytest
from unittest.mock import AsyncMock, MagicMock
from agent_customer_support.agents.coordinator import Coordinator
from agent_customer_support.models import (
    AgentResult, CustomerProfile, SessionState, Conversation,
)

pytestmark = pytest.mark.asyncio


def _coord():
    c = Coordinator()
    c.customers = AsyncMock()
    c.customers.get.return_value = CustomerProfile(customer_id="c1", name="C1",
                                                   enabled_modules=["m"])
    c.conversations = AsyncMock()
    c.conversations.load.return_value = Conversation(conversation_id="cv1",
                                                     customer_id="c1")
    c.sessions = AsyncMock()
    c.sessions.get.return_value = SessionState(conversation_id="cv1")
    c.rag = AsyncMock(); c.flow_store = AsyncMock()
    c.backlog = AsyncMock(); c.escalator = AsyncMock()
    # agent stubs
    c.guardrail = MagicMock()
    c.guardrail.check_input = AsyncMock(return_value={"pass": True, "reason": ""})
    c.guardrail.check_output = AsyncMock(return_value={"pass": True, "reason": ""})
    c.triage = MagicMock(); c.knowledge = MagicMock()
    c.flow = MagicMock(); c.verification = MagicMock(); c.escalation = MagicMock()
    return c


async def test_input_guardrail_block_short_circuits():
    c = _coord()
    c.guardrail.check_input = AsyncMock(return_value={"pass": False, "reason": "empty"})
    c.triage.run = AsyncMock()
    res = await c.handle_turn(customer_id="c1", conversation_id="cv1",
                              message="  ", attachments=[])
    assert res.escalated is False
    c.triage.run.assert_not_called()


async def test_triage_clarify_returns_question():
    c = _coord()
    c.triage.run = AsyncMock(return_value=AgentResult(action="reply",
                                                      reply="Bạn cần gì?"))
    res = await c.handle_turn(customer_id="c1", conversation_id="cv1",
                              message="?", attachments=[])
    assert res.reply == "Bạn cần gì?"


async def test_root_trace_uses_conversation_as_session(monkeypatch):
    from contextlib import contextmanager
    from agent_customer_support.agents import coordinator as coord_mod

    captured = {}

    @contextmanager
    def fake_trace(name, *, session_id=None, user_id=None, tags=None,
                   input=None, metadata=None):
        captured["session_id"] = session_id
        captured["user_id"] = user_id
        yield MagicMock()

    monkeypatch.setattr(coord_mod.tracing, "trace", fake_trace)
    c = _coord()
    c.triage.run = AsyncMock(return_value=AgentResult(action="reply", reply="ok"))
    await c.handle_turn(customer_id="c1", conversation_id="cv1",
                        message="?", attachments=[])
    assert captured["session_id"] == "cv1"
    assert captured["user_id"] == "c1"


async def test_knowledge_resolved_returns_reply():
    c = _coord()
    c.triage.run = AsyncMock(return_value=AgentResult(action="route",
                                                      routed_to="knowledge"))
    c.knowledge.run = AsyncMock(return_value=AgentResult(reply="đáp án",
                                                         resolved=True))
    res = await c.handle_turn(customer_id="c1", conversation_id="cv1",
                              message="cách làm X", attachments=[])
    assert res.reply == "đáp án"


async def test_knowledge_unresolved_escalates():
    c = _coord()
    c.triage.run = AsyncMock(return_value=AgentResult(action="route",
                                                      routed_to="knowledge"))
    c.knowledge.run = AsyncMock(return_value=AgentResult(reply="", resolved=False))
    c.escalation.run = AsyncMock(return_value=AgentResult(reply="chuyển nhân viên",
                                                          escalated=True))
    res = await c.handle_turn(customer_id="c1", conversation_id="cv1",
                              message="lỗi lạ", attachments=[])
    assert res.escalated is True
    c.escalation.run.assert_awaited_once()


async def test_suspected_bug_starts_verification():
    c = _coord()
    c.triage.run = AsyncMock(return_value=AgentResult(action="route",
                                                      routed_to="knowledge"))
    c.knowledge.run = AsyncMock(return_value=AgentResult(
        reply="nghi lỗi", resolved=False, suspected_bug=True,
        evidence={"module": "m", "summary": "A lỗi"}))
    c.verification.run = AsyncMock(return_value=AgentResult(
        reply="gửi ảnh giúp mình", evidence_complete=False))
    res = await c.handle_turn(customer_id="c1", conversation_id="cv1",
                              message="A bị lỗi", attachments=[])
    assert "ảnh" in res.reply
    saved = c.sessions.save.call_args.args[0]
    assert saved.pending == "verify_issue"


async def test_pending_verification_resumes_and_escalates_when_complete():
    c = _coord()
    c.sessions.get.return_value = SessionState(
        conversation_id="cv1", pending="verify_issue",
        pending_context={"module": "m", "summary": "A lỗi"})
    c.verification.run = AsyncMock(return_value=AgentResult(
        reply="đã đủ", evidence_complete=True,
        evidence={"module": "m", "summary": "A lỗi", "has_image": True}))
    c.escalation.run = AsyncMock(return_value=AgentResult(reply="chuyển nhân viên",
                                                          escalated=True))
    res = await c.handle_turn(customer_id="c1", conversation_id="cv1",
                              message="đây là ảnh", attachments=[])
    assert res.escalated is True
    c.backlog.add.assert_awaited_once()
    saved = c.sessions.save.call_args.args[0]
    assert saved.pending is None

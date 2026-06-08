import pytest
from unittest.mock import AsyncMock, patch
from agent_customer_support.agent.core import AgentCore, parse_goto
from agent_customer_support.models import CustomerProfile, Flow, FlowStep, FlowTransition
pytestmark = pytest.mark.asyncio

def test_parse_goto_found():
    assert parse_goto("Bạn làm bước này nhé. [[goto:s2]]") == ("Bạn làm bước này nhé.", "s2")

def test_parse_goto_absent():
    assert parse_goto("không có marker") == ("không có marker", None)

async def _build_core(llm_outputs):
    """llm_outputs: list of dicts returned sequentially from complete_with_tools."""
    core = AgentCore()
    core.customers = AsyncMock()
    core.customers.get.return_value = CustomerProfile(customer_id="c1", name="C1", enabled_modules=["m"])
    core.conversations = AsyncMock()
    core.sessions = AsyncMock()
    core.rag = AsyncMock(); core.flow_store = AsyncMock()
    core.backlog = AsyncMock(); core.escalator = AsyncMock()
    from agent_customer_support.models import SessionState
    core.sessions.get.return_value = SessionState(conversation_id="cv1")
    return core

async def test_simple_text_answer():
    core = await _build_core(None)
    seq = [{"stop_reason": "end", "text": "Chào bạn", "tool_calls": [], "raw": None}]
    with patch("agent_customer_support.agent.core.complete_with_tools", side_effect=seq):
        reply = await core.handle_turn(customer_id="c1", conversation_id="cv1", user_msg="hi")
    assert reply.reply == "Chào bạn"
    core.conversations.append.assert_awaited()

async def test_tool_then_answer():
    core = await _build_core(None)
    core.rag.search.return_value = {"passages": ["P"], "citations": ["c#1"], "top_confidence": 0.8}
    seq = [
        {"stop_reason": "tool_use", "text": None,
         "tool_calls": [{"id": "t1", "name": "search_knowledge", "input": {"query": "x"}}], "raw": None},
        {"stop_reason": "end", "text": "Đáp án dựa trên tài liệu", "tool_calls": [], "raw": None},
    ]
    with patch("agent_customer_support.agent.core.complete_with_tools", side_effect=seq):
        reply = await core.handle_turn(customer_id="c1", conversation_id="cv1", user_msg="cách làm X")
    assert "Đáp án" in reply.reply
    core.rag.search.assert_awaited_once()

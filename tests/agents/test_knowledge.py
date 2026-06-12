import pytest
from unittest.mock import patch, AsyncMock
from agent_customer_support.agents.knowledge import (
    KnowledgeAgent, parse_markers, needs_grading,
)
from agent_customer_support.agents.context import TurnContext
from agent_customer_support.models import CustomerProfile, SessionState, Conversation

pytestmark = pytest.mark.asyncio


def _ctx(message="cách tạo phiếu?") -> TurnContext:
    return TurnContext(
        customer=CustomerProfile(customer_id="c1", name="C1", enabled_modules=["m"]),
        session=SessionState(conversation_id="cv1"),
        conversation=Conversation(conversation_id="cv1", customer_id="c1"),
        message=message,
        transcript=f"user: {message}",
        rag=AsyncMock(), backlog=AsyncMock(), flow_store=AsyncMock(),
    )


# ---- pure helpers ----

def test_needs_grading_medium_band_true():
    assert needs_grading(0.70, ["a" * 500]) is True


def test_needs_grading_high_long_passages_false():
    assert needs_grading(0.88, ["a" * 500]) is False


def test_needs_grading_high_but_short_passages_true():
    assert needs_grading(0.88, ["short"]) is True


def test_needs_grading_no_passages_false():
    assert needs_grading(0.70, []) is False


def test_needs_grading_low_false():
    assert needs_grading(0.30, ["a" * 500]) is False


def test_parse_markers_no_answer():
    clean, kind, mod = parse_markers("Không rõ. [[no_answer]]")
    assert kind == "no_answer" and "[[no_answer]]" not in clean


def test_parse_markers_suspected_bug():
    clean, kind, mod = parse_markers("Đáng lẽ chạy. [[suspected_bug:xet-nghiem]]")
    assert kind == "suspected_bug" and mod == "xet-nghiem"


def test_parse_markers_plain_answer():
    clean, kind, mod = parse_markers("Vào menu X.")
    assert kind is None and mod is None and clean == "Vào menu X."


# ---- pipeline branches ----

async def test_high_confidence_composes_answer():
    ctx = _ctx()
    ctx.rag.search.return_value = {
        "passages": ["x" * 500], "citations": ["c#1"], "top_confidence": 0.9,
    }
    with patch("agent_customer_support.agents.knowledge.complete_text",
               return_value="Vào menu X rồi tạo."):
        res = await KnowledgeAgent().run(ctx)
    assert res.resolved is True
    assert "menu X" in res.reply
    ctx.rag.search.assert_awaited_once()  # no reformulation needed


async def test_medium_band_grader_present_then_answer():
    ctx = _ctx("thuật ngữ riêng của cty")
    ctx.rag.search.return_value = {
        "passages": ["p" * 500], "citations": [], "top_confidence": 0.70,
    }
    with patch("agent_customer_support.agents.knowledge.KnowledgeAgent._grade",
               new=AsyncMock(return_value=True)), \
         patch("agent_customer_support.agents.knowledge.complete_text",
               return_value="Trong phần mềm gọi là Y, làm thế này."):
        res = await KnowledgeAgent().run(ctx)
    assert res.resolved is True
    assert "Y" in res.reply


async def test_no_answer_marker_reformulates_then_logs():
    ctx = _ctx("hỏi linh tinh")
    ctx.rag.search.return_value = {
        "passages": ["p" * 500], "citations": [], "top_confidence": 0.90,
    }
    # compose returns no_answer both times -> reformulate once -> log_request
    with patch("agent_customer_support.agents.knowledge.KnowledgeAgent._reformulate",
               new=AsyncMock(return_value="reworded")), \
         patch("agent_customer_support.agents.knowledge.complete_text",
               return_value="Không có. [[no_answer]]"):
        res = await KnowledgeAgent().run(ctx)
    assert res.resolved is False
    assert ctx.rag.search.await_count == 2          # original + reformulated
    ctx.backlog.add.assert_awaited_once()
    assert ctx.backlog.add.call_args.kwargs["type"] == "how_to_missing"


async def test_suspected_bug_marker_sets_flag():
    ctx = _ctx("tính năng A bị lỗi")
    ctx.rag.search.return_value = {
        "passages": ["p" * 500], "citations": [], "top_confidence": 0.90,
    }
    with patch("agent_customer_support.agents.knowledge.complete_text",
               return_value="Đáng lẽ chạy. [[suspected_bug:xet-nghiem]]"):
        res = await KnowledgeAgent().run(ctx)
    assert res.suspected_bug is True
    assert res.evidence["module"] == "xet-nghiem"
    assert "[[suspected_bug" not in res.reply

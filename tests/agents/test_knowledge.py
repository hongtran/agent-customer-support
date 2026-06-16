import pytest
from unittest.mock import patch, AsyncMock, ANY
from agent_customer_support.agents.knowledge import (
    KnowledgeAgent,
    parse_markers,
    needs_grading,
)
from agent_customer_support.agents.context import TurnContext
from agent_customer_support.config import get_settings
from agent_customer_support.models import CustomerProfile, SessionState, Conversation

pytestmark = pytest.mark.asyncio


def _ctx(message="cách tạo phiếu?") -> TurnContext:
    return TurnContext(
        customer=CustomerProfile(customer_id="c1", name="C1", enabled_modules=["m"]),
        session=SessionState(conversation_id="cv1"),
        conversation=Conversation(conversation_id="cv1", customer_id="c1"),
        message=message,
        transcript=f"user: {message}",
        rag=AsyncMock(),
        backlog=AsyncMock(),
        flow_store=AsyncMock(),
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
        "passages": ["x" * 500],
        "citations": ["c#1"],
        "top_confidence": 0.9,
    }
    with patch(
        "agent_customer_support.agents.knowledge.complete_text", return_value="Vào menu X rồi tạo."
    ):
        res = await KnowledgeAgent().run(ctx)
    assert res.resolved is True
    assert "menu X" in res.reply
    ctx.rag.search.assert_awaited_once()  # no reformulation needed


async def test_medium_band_grader_present_then_answer():
    ctx = _ctx("thuật ngữ riêng của cty")
    ctx.rag.search.return_value = {
        "passages": ["p" * 500],
        "citations": [],
        "top_confidence": 0.70,
    }
    with (
        patch(
            "agent_customer_support.agents.knowledge.KnowledgeAgent._grade",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "agent_customer_support.agents.knowledge.complete_text",
            return_value="Trong phần mềm gọi là Y, làm thế này.",
        ),
    ):
        res = await KnowledgeAgent().run(ctx)
    assert res.resolved is True
    assert "Y" in res.reply


async def test_first_no_answer_clarifies_and_sets_pending():
    """First miss asks one clarifying question instead of logging/escalating."""
    ctx = _ctx("hỏi linh tinh")
    ctx.rag.search.return_value = {
        "passages": ["p" * 500],
        "citations": [],
        "top_confidence": 0.90,
    }
    with patch(
        "agent_customer_support.agents.knowledge.complete_text", return_value="[[no_answer]]"
    ):
        res = await KnowledgeAgent().run(ctx)
    assert res.resolved is None  # not escalated yet
    assert ctx.session.pending == "knowledge_clarify"
    ctx.backlog.add.assert_not_awaited()  # nothing logged on first miss


async def test_second_no_answer_logs_to_backlog():
    """Second miss (the clarification turn) gives up: log to backlog, clear pending."""
    ctx = _ctx("hỏi linh tinh")
    ctx.session.pending = "knowledge_clarify"  # we already clarified once
    ctx.rag.search.return_value = {
        "passages": ["p" * 500],
        "citations": [],
        "top_confidence": 0.90,
    }
    # compose returns bare [[no_answer]] (no substantial content) -> log to backlog
    with patch(
        "agent_customer_support.agents.knowledge.complete_text", return_value="[[no_answer]]"
    ):
        res = await KnowledgeAgent().run(ctx)
    assert res.resolved is False
    assert ctx.session.pending is None  # flag consumed
    assert ctx.rag.search.await_count == 1  # single attempt only
    ctx.backlog.add.assert_awaited_once()
    assert ctx.backlog.add.call_args.kwargs["type"] == "how_to_missing"


async def test_suspected_bug_marker_sets_flag():
    ctx = _ctx("tính năng A bị lỗi")
    ctx.rag.search.return_value = {
        "passages": ["p" * 500],
        "citations": [],
        "top_confidence": 0.90,
    }
    with patch(
        "agent_customer_support.agents.knowledge.complete_text",
        return_value="Đáng lẽ chạy. [[suspected_bug:xet-nghiem]]",
    ):
        res = await KnowledgeAgent().run(ctx)
    assert res.suspected_bug is True
    assert res.evidence["module"] == "xet-nghiem"
    assert "[[suspected_bug" not in res.reply


# ---- contextualize ----


async def test_contextualize_skips_on_first_turn():
    """No prior assistant turn → return ctx.message unchanged, no LLM call."""
    ctx = _ctx("cách tạo phiếu?")
    # transcript has no "assistant:" so _contextualize must return ctx.message directly
    with patch("agent_customer_support.agents.knowledge.complete_text") as mock_llm:
        result = await KnowledgeAgent()._contextualize(ctx, get_settings())
    assert result == ctx.message
    mock_llm.assert_not_called()


async def test_contextualize_resolves_pronouns_on_followup():
    """Prior assistant turn present → call LLM to produce standalone question."""
    ctx = _ctx("xoá nó thì sao?")
    ctx.transcript = (
        "user: cách tạo mẫu xét nghiệm?\n"
        "assistant: Vào menu Mẫu XN, nhấn Thêm.\n"
        "user: xoá nó thì sao?"
    )
    with patch(
        "agent_customer_support.agents.knowledge.complete_text",
        return_value="Cách xoá mẫu xét nghiệm trong CenLab?",
    ) as mock_llm:
        result = await KnowledgeAgent()._contextualize(ctx, get_settings())
    assert result == "Cách xoá mẫu xét nghiệm trong CenLab?"
    mock_llm.assert_called_once()


async def test_run_uses_contextualized_query_for_search():
    """On a follow-up turn the contextualized query (not raw ctx.message) hits RAG."""
    ctx = _ctx("xoá nó thì sao?")
    ctx.transcript = (
        "user: cách tạo mẫu xét nghiệm?\n"
        "assistant: Vào menu Mẫu XN, nhấn Thêm.\n"
        "user: xoá nó thì sao?"
    )
    ctx.rag.search.return_value = {
        "passages": ["x" * 500],
        "citations": [],
        "top_confidence": 0.9,
    }
    standalone = "Cách xoá mẫu xét nghiệm trong CenLab?"
    call_log: list[str] = []

    def fake_complete_text(**kwargs):
        content = kwargs["messages"][0]["content"]
        # _compose call — identified by the passages section it always includes
        if "Đoạn trích" in content:
            call_log.append(content)
            return "Vào menu Mẫu XN, chọn mẫu rồi nhấn Xoá."
        # _contextualize call
        return standalone

    with patch(
        "agent_customer_support.agents.knowledge.complete_text", side_effect=fake_complete_text
    ):
        res = await KnowledgeAgent().run(ctx)

    assert res.resolved is True
    ctx.rag.search.assert_awaited_once_with(standalone, collection=ANY)
    assert standalone in call_log[0]  # compose received the standalone question


async def test_compose_includes_history_on_followup():
    """_compose embeds 'Lịch sử hội thoại' section when there is prior context."""
    agent = KnowledgeAgent()
    transcript = (
        "user: cách tạo mẫu xét nghiệm?\n"
        "assistant: Vào menu Mẫu XN, nhấn Thêm.\n"
        "user: xoá nó thì sao?"
    )
    captured: dict = {}

    def fake_complete(*, messages, system, model=None):
        captured["content"] = messages[0]["content"]
        return "Chọn mẫu rồi nhấn Xoá."

    with patch("agent_customer_support.agents.knowledge.complete_text", side_effect=fake_complete):
        await agent._compose(
            "Cách xoá mẫu xét nghiệm?", ["passage text"], transcript, get_settings()
        )

    assert "Lịch sử hội thoại" in captured["content"]
    assert "Cách xoá mẫu xét nghiệm?" in captured["content"]


async def test_compose_omits_history_on_first_turn():
    """_compose omits the history section when there are no prior assistant turns."""
    agent = KnowledgeAgent()
    captured: dict = {}

    def fake_complete(*, messages, system, model=None):
        captured["content"] = messages[0]["content"]
        return "Vào menu X."

    with patch("agent_customer_support.agents.knowledge.complete_text", side_effect=fake_complete):
        await agent._compose(
            "cách tạo phiếu?", ["passage"], "user: cách tạo phiếu?", get_settings()
        )

    assert "Lịch sử hội thoại" not in captured["content"]


# ---- diagnose ----


async def test_diagnose_matches_known_symptom():
    with patch("agent_customer_support.agents.knowledge.complete_text",
               return_value='{"rule_id": "no_permission"}'):
        rule = await KnowledgeAgent()._diagnose("tôi không có quyền vào menu này", get_settings())
    assert rule is not None and rule.id == "no_permission"


async def test_diagnose_returns_none_when_no_match():
    with patch("agent_customer_support.agents.knowledge.complete_text",
               return_value='{"rule_id": "none"}'):
        rule = await KnowledgeAgent()._diagnose("cách tạo phiếu yêu cầu?", get_settings())
    assert rule is None


async def test_diagnose_failclosed_on_malformed_json():
    with patch("agent_customer_support.agents.knowledge.complete_text",
               return_value="xin lỗi tôi không biết"):
        rule = await KnowledgeAgent()._diagnose("bất kỳ", get_settings())
    assert rule is None


async def test_diagnose_failclosed_on_unknown_id():
    with patch("agent_customer_support.agents.knowledge.complete_text",
               return_value='{"rule_id": "made_up_rule"}'):
        rule = await KnowledgeAgent()._diagnose("bất kỳ", get_settings())
    assert rule is None

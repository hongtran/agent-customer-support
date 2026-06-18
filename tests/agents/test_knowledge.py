import pytest
from unittest.mock import patch, AsyncMock, ANY
from agent_customer_support.agents.knowledge import (
    KnowledgeAgent,
    parse_markers,
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


def test_parse_markers_no_answer():
    clean, kind, mod = parse_markers("Không rõ. [[no_answer]]")
    assert kind == "no_answer" and "[[no_answer]]" not in clean


def test_parse_markers_suspected_bug():
    clean, kind, mod = parse_markers("Đáng lẽ chạy. [[suspected_bug:xet-nghiem]]")
    assert kind == "suspected_bug" and mod == "xet-nghiem"


def test_parse_markers_plain_answer():
    clean, kind, mod = parse_markers("Vào menu X.")
    assert kind is None and mod is None and clean == "Vào menu X."


def test_parse_markers_clarify():
    clean, kind, mod = parse_markers(
        "Bạn đang muốn tạo loại phiếu nào?\n- Báo giá\n- PYC\n- Phiếu kết quả [[clarify]]"
    )
    assert kind == "clarify"
    assert mod is None
    assert "[[clarify]]" not in clean
    assert "Báo giá" in clean  # grounded options survive


def test_parse_markers_bug_beats_clarify():
    # If the model emits both, suspected_bug wins (safe handoff path).
    clean, kind, mod = parse_markers("Đáng lẽ chạy. [[clarify]] [[suspected_bug:xn]]")
    assert kind == "suspected_bug" and mod == "xn"


def test_parse_markers_strips_stray_second_marker():
    # Model misbehaves and emits two markers; precedence picks clarify, but the
    # stray no_answer marker must not leak into the user-facing text.
    clean, kind, mod = parse_markers("Bạn muốn loại nào? [[no_answer]] [[clarify]]")
    assert kind == "clarify"
    assert "[[no_answer]]" not in clean
    assert "[[clarify]]" not in clean


def test_compose_prompt_documents_clarify_and_diagnose_policy():
    from agent_customer_support.agents.prompts import (
        KNOWLEDGE_COMPOSE_PROMPT,
        KNOWLEDGE_RESUME_NO_CLARIFY,
    )

    # The clarify/confirm contract must be in the compose system prompt...
    assert "[[clarify]]" in KNOWLEDGE_COMPOSE_PROMPT
    # ...the diagnose (process-conformance) contract too...
    assert "sai quy trình" in KNOWLEDGE_COMPOSE_PROMPT
    # ...and the resume suppressor must forbid re-clarifying.
    assert "[[clarify]]" in KNOWLEDGE_RESUME_NO_CLARIFY
    assert "KHÔNG" in KNOWLEDGE_RESUME_NO_CLARIFY


# ---- pipeline branches ----


async def test_composes_answer_from_passages():
    ctx = _ctx()
    ctx.rag.search.return_value = {"passages": ["x" * 500], "citations": ["c#1"]}
    with patch(
        "agent_customer_support.agents.knowledge.complete_text", return_value="Vào menu X rồi tạo."
    ):
        res = await KnowledgeAgent().run(ctx)
    assert res.resolved is True
    assert "menu X" in res.reply
    ctx.rag.search.assert_awaited_once()


async def test_always_composes_even_with_empty_passages():
    """Process is always-on, so a process-level question is answerable with no RAG hits."""
    ctx = _ctx("ai phụ trách bước nghiệm thu?")
    ctx.rag.search.return_value = {"passages": [], "citations": []}
    with patch(
        "agent_customer_support.agents.knowledge.complete_text",
        return_value="Nghiệm thu hợp đồng do Kế toán và Kinh doanh phụ trách.",
    ) as mock_llm:
        res = await KnowledgeAgent().run(ctx)
    assert res.resolved is True
    mock_llm.assert_called_once()  # compose runs even with no passages


async def test_first_no_answer_clarifies_and_sets_pending():
    """First miss asks one clarifying question instead of logging/escalating."""
    ctx = _ctx("hỏi linh tinh")
    ctx.rag.search.return_value = {"passages": [], "citations": []}
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
    ctx.rag.search.return_value = {"passages": [], "citations": []}
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
    ctx.rag.search.return_value = {"passages": ["p" * 500], "citations": []}
    with patch(
        "agent_customer_support.agents.knowledge.complete_text",
        return_value="Đáng lẽ chạy. [[suspected_bug:xet-nghiem]]",
    ):
        res = await KnowledgeAgent().run(ctx)
    assert res.suspected_bug is True
    assert res.evidence["application"] == "xet-nghiem"
    assert "[[suspected_bug" not in res.reply


@pytest.mark.parametrize(
    "message, clarify_reply",
    [
        # ambiguous subject
        (
            "cách tạo phiếu?",
            "Bạn muốn tạo loại phiếu nào?\n- Báo giá\n- PYC\n- Phiếu kết quả [[clarify]]",
        ),
        # missing decisive parameter / unknown user-state
        (
            "PQT trả đơn về thì KD sửa số lượng mẫu được không?",
            "Đơn của bạn đang ở trạng thái nào?\n- Còn trong ứng dụng\n"
            "- Đã chuyển chưa tiếp nhận\n- Đã tiếp nhận ở ứng dụng khác [[clarify]]",
        ),
        # unverified premise
        (
            "sau khi huỷ PYC thì hoàn tiền thế nào?",
            "Bạn đã thực sự huỷ PYC chưa, hay đang cân nhắc? [[clarify]]",
        ),
    ],
)
async def test_clarify_asks_once_and_sets_pending(message, clarify_reply):
    ctx = _ctx(message)
    ctx.rag.search.return_value = {"passages": [], "citations": ["c#1"]}
    with patch("agent_customer_support.agents.knowledge.complete_text", return_value=clarify_reply):
        res = await KnowledgeAgent().run(ctx)
    assert res.resolved is None  # neither answered nor escalated
    assert ctx.session.pending == "knowledge_clarify"
    assert "[[clarify]]" not in res.reply
    ctx.backlog.add.assert_not_awaited()  # clarify is not a miss


async def test_resume_turn_disables_clarify_and_grounds_answer():
    """On resume, compose is called with allow_clarify=False; a grounded answer returns."""
    ctx = _ctx("đơn đang còn trong ứng dụng")
    ctx.session.pending = "knowledge_clarify"  # we clarified last turn
    ctx.rag.search.return_value = {"passages": ["p" * 200], "citations": []}
    seen: dict = {}

    def fake_complete(**kwargs):
        content = kwargs["messages"][0]["content"]
        if "Đoạn trích" in content:  # the compose call
            seen["compose_content"] = content
            return "Vì đơn còn trong ứng dụng, bạn trả về tài khoản đã tạo để sửa."
        return kwargs["messages"][0]["content"]  # contextualize passthrough

    with patch("agent_customer_support.agents.knowledge.complete_text", side_effect=fake_complete):
        res = await KnowledgeAgent().run(ctx)

    from agent_customer_support.agents.prompts import KNOWLEDGE_RESUME_NO_CLARIFY

    assert res.resolved is True
    assert ctx.session.pending is None  # flag consumed
    assert KNOWLEDGE_RESUME_NO_CLARIFY in seen["compose_content"]  # clarify suppressed
    ctx.backlog.add.assert_not_awaited()


async def test_clarify_marker_on_resume_is_downgraded_to_answer():
    """Defensive: if the model disobeys and re-emits [[clarify]] on resume, answer anyway."""
    ctx = _ctx("vẫn chưa rõ")
    ctx.session.pending = "knowledge_clarify"
    ctx.rag.search.return_value = {"passages": [], "citations": []}
    with patch(
        "agent_customer_support.agents.knowledge.complete_text",
        return_value="Giả định đơn còn trong ứng dụng: bạn sửa trực tiếp. [[clarify]]",
    ):
        res = await KnowledgeAgent().run(ctx)
    assert res.resolved is True  # not a second clarify
    assert ctx.session.pending is None
    assert "[[clarify]]" not in res.reply


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
    ctx.rag.search.return_value = {"passages": ["x" * 500], "citations": []}
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
    ctx.rag.search.assert_awaited_once_with(standalone, collection=ANY, applications=None)
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


async def test_compose_appends_no_clarify_directive_when_disabled():
    from agent_customer_support.agents.prompts import KNOWLEDGE_RESUME_NO_CLARIFY

    agent = KnowledgeAgent()
    captured: dict = {}

    def fake_complete(*, messages, system, model=None):
        captured["content"] = messages[0]["content"]
        return "ok"

    with patch("agent_customer_support.agents.knowledge.complete_text", side_effect=fake_complete):
        await agent._compose("q", ["p"], "user: q", get_settings(), allow_clarify=False)

    assert KNOWLEDGE_RESUME_NO_CLARIFY in captured["content"]


async def test_compose_omits_no_clarify_directive_by_default():
    from agent_customer_support.agents.prompts import KNOWLEDGE_RESUME_NO_CLARIFY

    agent = KnowledgeAgent()
    captured: dict = {}

    def fake_complete(*, messages, system, model=None):
        captured["content"] = messages[0]["content"]
        return "ok"

    with patch("agent_customer_support.agents.knowledge.complete_text", side_effect=fake_complete):
        await agent._compose("q", ["p"], "user: q", get_settings())

    assert KNOWLEDGE_RESUME_NO_CLARIFY not in captured["content"]


async def test_compose_passes_process_block_as_cached_system_prefix():
    """The always-on process context must be the first (cacheable) system block."""
    from agent_customer_support.agents.prompts import PROCESS_BLOCK

    agent = KnowledgeAgent()
    captured: dict = {}

    def fake_complete(*, messages, system, model=None):
        captured["system"] = system
        return "ok"

    with patch("agent_customer_support.agents.knowledge.complete_text", side_effect=fake_complete):
        await agent._compose("q", ["p"], "user: q", get_settings())

    assert isinstance(captured["system"], list)
    assert captured["system"][0] is PROCESS_BLOCK

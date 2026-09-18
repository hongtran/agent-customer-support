import pytest
from unittest.mock import patch, AsyncMock, ANY
from agent_customer_support.agents.knowledge import (
    KnowledgeAgent,
    parse_markers,
)
from agent_customer_support.agents.context import TurnContext
from agent_customer_support.config import get_settings
from agent_customer_support.llm.schemas import CitedSource, ComposedAnswer
from agent_customer_support.models import CustomerProfile, SessionState, Conversation

pytestmark = pytest.mark.asyncio


def _sources(*ids: str) -> list[CitedSource]:
    """Citations by id with no section — the shape for a passage that offers no heading.

    Tests about sections build CitedSource directly; everything else only cares that a
    source was declared, so it should not have to say anything about headings.
    """
    return [CitedSource(id=i, section="") for i in ids]


def _composed(answer: str, cited: list[str] | list[CitedSource] | None = None):
    """Patch the compose call.

    Compose is the only structured call KnowledgeAgent makes; contextualize still goes
    through `complete_text`, so the two are patched separately and a test that cares
    about only one of them no longer has to disambiguate by prompt content.

    `cited` accepts bare ids for convenience, or CitedSource when the section matters.
    """
    sources = [CitedSource(id=c, section="") if isinstance(c, str) else c for c in cited or []]
    return patch(
        "agent_customer_support.agents.knowledge.complete_structured",
        return_value=ComposedAnswer(answer=answer, cited=sources),
    )


def _ctx(message="cách tạo phiếu?") -> TurnContext:
    ctx = TurnContext(
        customer=CustomerProfile(customer_id="c1", name="C1", enabled_modules=["m"]),
        session=SessionState(conversation_id="cv1"),
        conversation=Conversation(conversation_id="cv1", customer_id="c1"),
        message=message,
        transcript=f"user: {message}",
        rag=AsyncMock(),
        backlog=AsyncMock(),
        flow_store=AsyncMock(),
        qa_store=AsyncMock(),
    )
    # The product search goes through search_with_fallback and the Q&A search through
    # search. Aliasing them to one mock keeps `search.return_value` covering both, which
    # is what these tests assume, and makes `await_count` the total round-trip count.
    ctx.rag.search_with_fallback = ctx.rag.search
    return ctx


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
    with _composed("Vào menu X rồi tạo.", cited=["0"]):
        res = await KnowledgeAgent().run(ctx)
    assert res.resolved is True
    assert "menu X" in res.reply
    assert ctx.rag.search.await_count >= 1


async def test_always_composes_even_with_empty_passages():
    """Process is always-on, so a process-level question is answerable with no RAG hits."""
    ctx = _ctx("ai phụ trách bước nghiệm thu?")
    ctx.rag.search.return_value = {"passages": [], "citations": []}
    with _composed(
        "Nghiệm thu hợp đồng do Kế toán và Kinh doanh phụ trách.", cited=["quy_trinh_chung"]
    ) as mock_llm:
        res = await KnowledgeAgent().run(ctx)
    assert res.resolved is True
    mock_llm.assert_called_once()  # compose runs even with no passages


async def test_first_no_answer_clarifies_and_sets_pending():
    """First miss asks one clarifying question instead of logging/escalating."""
    ctx = _ctx("hỏi linh tinh")
    ctx.rag.search.return_value = {"passages": [], "citations": []}
    with _composed("[[no_answer]]"):
        res = await KnowledgeAgent().run(ctx)
    assert res.resolved is None  # not escalated yet
    assert ctx.session.pending == "knowledge_clarify"
    ctx.backlog.add.assert_not_awaited()  # nothing logged on first miss


async def test_second_no_answer_logs_to_backlog():
    """Second miss (the clarification turn) gives up: log to backlog, clear pending."""
    ctx = _ctx("hỏi linh tinh")
    ctx.session.pending = "knowledge_clarify"  # we already clarified once
    ctx.rag.search.return_value = {"passages": [], "citations": []}
    with _composed("[[no_answer]]"):
        res = await KnowledgeAgent().run(ctx)
    assert res.resolved is False
    assert ctx.session.pending is None  # flag consumed
    assert ctx.rag.search.await_count == 2  # product + qa (one each, single round-trip)
    ctx.backlog.add.assert_awaited_once()
    assert ctx.backlog.add.call_args.kwargs["type"] == "how_to_missing"


async def test_suspected_bug_marker_sets_flag():
    ctx = _ctx("tính năng A bị lỗi")
    ctx.rag.search.return_value = {"passages": ["p" * 500], "citations": []}
    with _composed("Đáng lẽ chạy. [[suspected_bug:xet-nghiem]]", cited=["0"]):
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
    with _composed(clarify_reply):
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

    def fake_compose(**kwargs):
        seen["compose_content"] = kwargs["messages"][0]["content"]
        return ComposedAnswer(
            answer="Vì đơn còn trong ứng dụng, bạn trả về tài khoản đã tạo để sửa.",
            cited=_sources("0"),
        )

    with patch(
        "agent_customer_support.agents.knowledge.complete_structured", side_effect=fake_compose
    ):
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
    with _composed("Giả định đơn còn trong ứng dụng: bạn sửa trực tiếp. [[clarify]]"):
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

    def fake_compose(**kwargs):
        call_log.append(kwargs["messages"][0]["content"])
        return ComposedAnswer(answer="Vào menu Mẫu XN, chọn mẫu rồi nhấn Xoá.", cited=_sources("0"))

    with (
        patch("agent_customer_support.agents.knowledge.complete_text", return_value=standalone),
        patch(
            "agent_customer_support.agents.knowledge.complete_structured",
            side_effect=fake_compose,
        ),
    ):
        res = await KnowledgeAgent().run(ctx)

    assert res.resolved is True
    ctx.rag.search.assert_any_await(
        standalone, collection=ANY, applications=None, fallback_applications=None
    )
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

    def fake_compose(*, messages, system, model=None, schema=None):
        captured["content"] = messages[0]["content"]
        return ComposedAnswer(answer="Chọn mẫu rồi nhấn Xoá.", cited=[])

    with patch(
        "agent_customer_support.agents.knowledge.complete_structured", side_effect=fake_compose
    ):
        await agent._compose(
            "Cách xoá mẫu xét nghiệm?", ["passage text"], transcript, get_settings()
        )

    assert "Lịch sử hội thoại" in captured["content"]
    assert "Cách xoá mẫu xét nghiệm?" in captured["content"]


async def test_compose_omits_history_on_first_turn():
    """_compose omits the history section when there are no prior assistant turns."""
    agent = KnowledgeAgent()
    captured: dict = {}

    def fake_compose(*, messages, system, model=None, schema=None):
        captured["content"] = messages[0]["content"]
        return ComposedAnswer(answer="Vào menu X.", cited=[])

    with patch(
        "agent_customer_support.agents.knowledge.complete_structured", side_effect=fake_compose
    ):
        await agent._compose(
            "cách tạo phiếu?", ["passage"], "user: cách tạo phiếu?", get_settings()
        )

    assert "Lịch sử hội thoại" not in captured["content"]


async def test_compose_appends_no_clarify_directive_when_disabled():
    from agent_customer_support.agents.prompts import KNOWLEDGE_RESUME_NO_CLARIFY

    agent = KnowledgeAgent()
    captured: dict = {}

    def fake_compose(*, messages, system, model=None, schema=None):
        captured["content"] = messages[0]["content"]
        return ComposedAnswer(answer="ok", cited=[])

    with patch(
        "agent_customer_support.agents.knowledge.complete_structured", side_effect=fake_compose
    ):
        await agent._compose("q", ["p"], "user: q", get_settings(), allow_clarify=False)

    assert KNOWLEDGE_RESUME_NO_CLARIFY in captured["content"]


async def test_compose_omits_no_clarify_directive_by_default():
    from agent_customer_support.agents.prompts import KNOWLEDGE_RESUME_NO_CLARIFY

    agent = KnowledgeAgent()
    captured: dict = {}

    def fake_compose(*, messages, system, model=None, schema=None):
        captured["content"] = messages[0]["content"]
        return ComposedAnswer(answer="ok", cited=[])

    with patch(
        "agent_customer_support.agents.knowledge.complete_structured", side_effect=fake_compose
    ):
        await agent._compose("q", ["p"], "user: q", get_settings())

    assert KNOWLEDGE_RESUME_NO_CLARIFY not in captured["content"]


async def test_compose_passes_process_block_as_cached_system_prefix():
    """The always-on process context must be the first (cacheable) system block."""
    from agent_customer_support.agents.prompts import PROCESS_BLOCK

    agent = KnowledgeAgent()
    captured: dict = {}

    def fake_compose(*, messages, system, model=None, schema=None):
        captured["system"] = system
        return ComposedAnswer(answer="ok", cited=[])

    with patch(
        "agent_customer_support.agents.knowledge.complete_structured", side_effect=fake_compose
    ):
        await agent._compose("q", ["p"], "user: q", get_settings())

    assert isinstance(captured["system"], list)
    assert captured["system"][0] is PROCESS_BLOCK


# ---- wrong application selected: widened retry + telling the user which module ----


def test_other_applications_names_only_the_modules_outside_the_selection():
    from agent_customer_support.agents.knowledge import _other_applications

    metas = [
        {"application": "phong_thi_nghiem"},
        {"application": "mua_sam"},  # the user's own selection — not a mismatch
        {"confidence": 0.9},  # global document: belongs to no module
        {"application": "phong_thi_nghiem"},  # duplicate
    ]
    # Selection arrives as a display name and the metas carry slugs; comparing the two
    # forms directly would report the user's own module back at them.
    assert _other_applications(metas, ["Mua sắm"]) == ["Phòng thí nghiệm"]
    # An unmapped slug still renders — better a raw slug than a silently dropped module.
    assert _other_applications([{"application": "khong_biet"}], None) == ["khong_biet"]
    assert _other_applications([], ["Mua sắm"]) == []


async def _run_capturing_compose(ctx, search_result):
    """Run the agent against one product-search result, returning the compose prompt."""
    ctx.rag.search_with_fallback = AsyncMock(return_value=search_result)
    ctx.rag.search = AsyncMock(return_value={"passages": [], "citations": []})  # qa
    composed = []

    def fake_compose(**kwargs):
        composed.append(kwargs["messages"][0]["content"])
        return ComposedAnswer(
            answer="Anh/Chị vui lòng vào menu Mẫu XN." + "x" * 200, cited=_sources("0")
        )

    with (
        patch("agent_customer_support.agents.knowledge.complete_text", return_value=ctx.message),
        patch(
            "agent_customer_support.agents.knowledge.complete_structured",
            side_effect=fake_compose,
        ),
    ):
        res = await KnowledgeAgent().run(ctx)
    return res, composed[0]


async def test_search_is_widened_to_every_module_the_customer_is_entitled_to():
    ctx = _ctx()
    ctx.customer.enabled_applications = ["Mua sắm", "Phòng thí nghiệm"]
    ctx.session.selected_applications = ["Mua sắm"]
    await _run_capturing_compose(ctx, {"passages": ["p"], "citations": [], "metas": []})

    kwargs = ctx.rag.search_with_fallback.await_args.kwargs
    assert kwargs["applications"] == ["Mua sắm"]
    # Never wider than the entitlement: a customer must not be told about a module
    # they did not buy and cannot see in their UI.
    assert kwargs["fallback_applications"] == ["Mua sắm", "Phòng thí nghiệm"]


async def test_a_widened_hit_tells_the_composer_which_module_it_came_from():
    ctx = _ctx()
    ctx.customer.enabled_applications = ["Mua sắm", "Phòng thí nghiệm"]
    ctx.session.selected_applications = ["Mua sắm"]
    res, prompt = await _run_capturing_compose(
        ctx,
        {
            "passages": ["p"],
            "citations": ["lab#1"],
            "metas": [{"application": "phong_thi_nghiem", "confidence": 0.9}],
            "fallback_used": True,
        },
    )
    assert res.resolved is True
    assert "LƯU Ý PHẠM VI" in prompt
    assert "Phòng thí nghiệm" in prompt


async def test_a_normal_hit_says_nothing_about_scope():
    """The note costs prompt tokens and risks a confusing aside, so it appears only
    when retrieval actually had to leave the user's selected module."""
    ctx = _ctx()
    ctx.session.selected_applications = ["Mua sắm"]
    _, prompt = await _run_capturing_compose(
        ctx,
        {
            "passages": ["p"],
            "citations": [],
            "metas": [{"application": "mua_sam", "confidence": 0.9}],
            "fallback_used": False,
        },
    )
    assert "LƯU Ý PHẠM VI" not in prompt


async def test_a_widened_hit_on_a_global_document_names_no_module():
    """Untagged documents are global by design; there is no module to point the user at."""
    ctx = _ctx()
    ctx.session.selected_applications = ["Mua sắm"]
    _, prompt = await _run_capturing_compose(
        ctx,
        {
            "passages": ["p"],
            "citations": [],
            "metas": [{"confidence": 0.9}],
            "fallback_used": True,
        },
    )
    assert "LƯU Ý PHẠM VI" not in prompt


async def test_the_scope_note_renders_both_module_lists():
    """Pins the note's placeholders against its call site. `str.format` resolves every
    placeholder on each call, so a renamed or newly-added one raises KeyError mid-turn
    — after the retrieval spend, on exactly the path that was supposed to rescue the
    answer. Asserting the rendered text is what makes that a test failure instead."""
    ctx = _ctx()
    ctx.customer.enabled_applications = ["Mua sắm", "Phòng thí nghiệm", "Quản lý kho"]
    ctx.session.selected_applications = ["Mua sắm", "Quản lý kho"]
    _, prompt = await _run_capturing_compose(
        ctx,
        {
            "passages": ["p"],
            "citations": [],
            "metas": [{"application": "phong_thi_nghiem", "confidence": 0.9}],
            "fallback_used": True,
        },
    )
    note = prompt.split("LƯU Ý PHẠM VI")[1]
    assert "{" not in note, "an unfilled placeholder leaked into the prompt"
    assert "Mua sắm, Quản lý kho" in note  # what the user selected
    assert "Phòng thí nghiệm" in note  # where the passages actually came from


async def test_no_scope_note_without_a_selection_to_contrast_against():
    """A widened retry is impossible with no selection (an empty scope is already
    global), but the note must not half-render if a caller ever gets there."""
    agent = KnowledgeAgent()
    with _composed("ok") as llm:
        await agent._compose(
            "q",
            ["p"],
            "",
            get_settings(),
            other_applications=["Phòng thí nghiệm"],
            selected_applications=None,
        )
    assert "LƯU Ý PHẠM VI" not in llm.call_args.kwargs["messages"][0]["content"]


# ---- citations: what the answer declared, validated against what was retrieved ----


async def _run_with_cited(cited, metas=None):
    ctx = _ctx()
    ctx.rag.search_with_fallback = AsyncMock(
        return_value={
            "passages": ["p0", "p1"],
            "citations": ["ignored"],
            "metas": metas
            if metas is not None
            else [
                {"doc_id": "d1", "url": "chunks/9. HDSD - Lấy mẫu.docx", "confidence": 0.9},
                {"doc_id": "d2", "url": "chunks/2. HDSD - Mua sắm.docx", "confidence": 0.8},
            ],
        }
    )
    ctx.rag.search = AsyncMock(return_value={"passages": [], "citations": [], "metas": []})
    with _composed("Anh/Chị vui lòng vào menu X." + "y" * 200, cited=cited):
        return await KnowledgeAgent().run(ctx)


async def test_only_declared_sources_are_cited():
    """Not everything retrieved. Two passages came back and the answer used one."""
    res = await _run_with_cited(["1"])
    assert [c.doc_id for c in res.citations] == ["d2"]
    assert res.cited_passages == ["p1"]


async def test_an_invented_index_never_reaches_the_result():
    """The catalog is the guard, exactly as it is for image markers. A citation the
    user cannot verify is worse than no citation at all."""
    res = await _run_with_cited(["0", "9"])
    assert [c.doc_id for c in res.citations] == ["d1"]


async def test_a_process_only_answer_shows_no_source_at_all():
    """The process block is our system prompt, not a document the customer can open, so
    an answer resting only on it stays quiet rather than naming something unresolvable.
    The declaration still happens — it just does not reach the reply."""
    res = await _run_with_cited(["quy_trinh_chung"])
    assert res.citations == []
    # Empty cited_passages is also what makes the guardrail skip this turn: there is no
    # passage to judge a process answer against.
    assert res.cited_passages == []


async def test_a_clarify_reply_cites_nothing_it_did_not_use():
    ctx = _ctx()
    ctx.rag.search.return_value = {"passages": ["p0"], "citations": ["c1"], "metas": []}
    with _composed("Bạn muốn tạo loại phiếu nào? [[clarify]]", cited=[]):
        res = await KnowledgeAgent().run(ctx)
    assert res.citations == []
    assert res.cited_passages == []


async def test_a_miss_carries_no_citations():
    """The reply on this path is our own canned text, not composed from any source."""
    ctx = _ctx("hỏi linh tinh")
    ctx.session.pending = "knowledge_clarify"
    ctx.rag.search.return_value = {
        "passages": ["p0"],
        "citations": ["c1"],
        "metas": [{"doc_id": "d1", "url": "x/9. HDSD.docx"}],
    }
    with _composed("[[no_answer]]", cited=["0"]):
        res = await KnowledgeAgent().run(ctx)
    assert res.resolved is False
    assert res.citations == []


async def test_structured_compose_failure_still_answers_uncited():
    """Constrained decoding produced nothing usable. Retrieval and contextualize are
    already paid for, so ship the answer without its source list rather than lose it."""
    ctx = _ctx()
    ctx.rag.search.return_value = {"passages": ["p0"], "citations": [], "metas": []}
    with (
        patch("agent_customer_support.agents.knowledge.complete_structured", return_value=None),
        patch(
            "agent_customer_support.agents.knowledge.complete_text",
            return_value="Anh/Chị vui lòng vào menu X." + "y" * 200,
        ) as fallback,
    ):
        res = await KnowledgeAgent().run(ctx)
    assert res.resolved is True
    assert "menu X" in res.reply
    assert res.citations == []
    fallback.assert_called_once()


# ---- the compose prompt offers the headings the composer must choose from ----


async def _compose_content(passages: list[str]) -> str:
    """The user content of one compose call, for asserting on what the model was shown."""
    captured: dict = {}

    def fake_compose(*, messages, system, model=None, schema=None):
        captured["content"] = messages[0]["content"]
        return ComposedAnswer(answer="ok", cited=[])

    with patch(
        "agent_customer_support.agents.knowledge.complete_structured", side_effect=fake_compose
    ):
        await KnowledgeAgent()._compose("q", passages, "user: q", get_settings())
    return captured["content"]


async def test_the_compose_prompt_lists_each_passage_s_real_headings():
    """Without this list the model reaches for whatever looks most like a title, which in
    this corpus is the summary line prepended to every chunk — not a heading at all."""
    passage = (
        "Quản lý, nhập ký hiệu và gán phép thử cho mẫu trong PYC\n\n"
        "Chunk này mô tả các thao tác quản lý mẫu.\n\n"
        "##### Thao tác với mẫu đã tạo\n\n- Sửa mẫu.\n\n"
        "##### **Import ký hiệu mẫu:**\n\n1. Mở chức năng Import."
    )
    content = await _compose_content([passage])
    assert "(các mục trong đoạn này: Thao tác với mẫu đã tạo | Import ký hiệu mẫu)" in content
    # The summary line is in the passage body but must not be offered as a choice.
    assert "này: Quản lý, nhập ký hiệu" not in content


async def test_a_passage_with_no_headings_gets_no_annotation():
    """An empty `(các mục trong đoạn này: )` would invite the model to fill it in."""
    content = await _compose_content(["| Vai trò | Không | Gán một hoặc nhiều vai trò. |"])
    assert "các mục trong đoạn này" not in content
    assert "[0]" in content  # still numbered as before


# ---- repair: rewrite an answer the guardrail flagged with only minor claims ----

_CLAIMS = [
    {"span": "ở góc phải", "severity": "minor", "reason": "Nguồn không nêu vị trí nút"},
]


async def test_repair_sends_sources_answer_and_claims_under_the_process_block():
    from agent_customer_support.agents.prompts import KNOWLEDGE_REPAIR_PROMPT, PROCESS_BLOCK

    captured: dict = {}

    def fake_text(*, messages, system, model=None):
        captured["system"] = system
        captured["content"] = messages[0]["content"]
        return "Nhấn Lưu để lưu phiếu."

    with patch("agent_customer_support.agents.knowledge.complete_text", side_effect=fake_text):
        out = await KnowledgeAgent().repair(
            "Nhấn Lưu ở góc phải để lưu phiếu.", _CLAIMS, ["Nhấn Lưu để lưu phiếu."]
        )

    assert out == "Nhấn Lưu để lưu phiếu."
    assert captured["system"][0] is PROCESS_BLOCK
    assert captured["system"][-1]["text"] == KNOWLEDGE_REPAIR_PROMPT
    assert "[0] Nhấn Lưu để lưu phiếu." in captured["content"]
    assert "Nhấn Lưu ở góc phải để lưu phiếu." in captured["content"]
    assert "ở góc phải" in captured["content"]
    assert "Nguồn không nêu vị trí nút" in captured["content"]
    assert "Xóa hoặc sửa các ý sau cho khớp với nguồn. Không thêm ý mới." in captured["content"]


async def test_repair_returns_none_when_the_model_produces_nothing():
    with patch("agent_customer_support.agents.knowledge.complete_text", return_value="  "):
        assert await KnowledgeAgent().repair("Nhấn Lưu.", _CLAIMS, ["p"]) is None


async def test_repair_drops_an_image_marker_the_original_did_not_have():
    """The prompt says add nothing, but an invented marker would be presigned into a
    broken image, so the guard is in code — the same rule as doc_images.select."""
    original = "Nhấn [[img:icon:lay_mau/image3.png]] ở góc phải để lưu."
    repaired = "Nhấn [[img:icon:lay_mau/image3.png]] để lưu. [[img:screen:lay_mau/image9.png]]"
    with patch("agent_customer_support.agents.knowledge.complete_text", return_value=repaired):
        out = await KnowledgeAgent().repair(original, _CLAIMS, ["p"])
    assert out == "Nhấn [[img:icon:lay_mau/image3.png]] để lưu."


async def test_repair_is_labelled_as_its_own_llm_step():
    """Lands in Langfuse as llm.knowledge.repair, so an evaluator can target it apart
    from the compose call."""
    from agent_customer_support.observability import tracing

    seen: dict = {}

    def fake_text(*, messages, system, model=None):
        seen["labels"] = tracing.current_labels()
        return "ok đã sửa"

    with patch("agent_customer_support.agents.knowledge.complete_text", side_effect=fake_text):
        with tracing.agent_span("knowledge"):
            await KnowledgeAgent().repair("Nhấn Lưu ở góc phải.", _CLAIMS, ["p"])
    assert seen["labels"] == ("knowledge", "repair")

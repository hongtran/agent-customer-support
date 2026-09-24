import pytest
from unittest.mock import patch, AsyncMock
from agent_customer_support.agents.issue_verification import IssueVerificationAgent, fallback_report
from agent_customer_support.llm.schemas import BugReport, VerificationDecision
from agent_customer_support.agents.context import TurnContext
from agent_customer_support.models import (
    Attachment,
    BugSlots,
    Conversation,
    CustomerProfile,
    SessionState,
    VerifyContext,
)

pytestmark = pytest.mark.asyncio


def _slots(**kw) -> BugSlots:
    """A model's reading of the slots, defaulting every one it did not fill."""
    return BugSlots.empty().model_copy(update=kw)


def _decision(outcome="need_more_info", reply="...", ask_for=None, **slots) -> VerificationDecision:
    return VerificationDecision(
        outcome=outcome, reply=reply, slots=_slots(**slots), ask_for=ask_for or []
    )


def _ctx(message, attachments=None, context=None, doc_checked=True) -> TurnContext:
    """A verification turn. `doc_checked` defaults to True — the slot-filling tests are
    about what happens after the doc check, so they skip it; the doc-check tests pass
    False."""
    context = (
        context
        if context is not None
        else VerifyContext(application="lay_mau_quan_trac", summary="A bị lỗi").model_dump()
    )
    s = SessionState(
        conversation_id="cv1",
        pending="verify_issue",
        pending_context={**context, "doc_checked": doc_checked},
    )
    return TurnContext(
        customer=CustomerProfile(customer_id="c1", name="C1"),
        session=s,
        conversation=Conversation(conversation_id="cv1", customer_id="c1"),
        message=message,
        attachments=attachments or [],
        transcript=f"user: {message}",
        rag=AsyncMock(),
        backlog=AsyncMock(),
        escalator=AsyncMock(),
    )


def _patch(decision, report=None):
    """Patch the two structured calls the agent makes, in the order it makes them."""
    return patch(
        "agent_customer_support.agents.issue_verification.complete_structured",
        side_effect=[decision, report],
    )


# ---- the three outcomes ----


async def test_incomplete_evidence_asks_for_more_and_files_nothing():
    with _patch(_decision(reply="Bạn gửi giúp mình ảnh lỗi nhé?")):
        res = await IssueVerificationAgent().run(_ctx("nó cứ lỗi thôi"))
    assert res.verify_outcome == "need_more_info"
    assert "ảnh" in res.reply
    # No report on this path: a ticket is only written up once the evidence is in.
    assert res.evidence["report"] is None


async def test_a_user_error_returns_an_explanation_and_no_report():
    with _patch(
        _decision(outcome="user_error", reply="Chức năng này cần chọn mẫu trước khi bấm Lưu.")
    ):
        res = await IssueVerificationAgent().run(_ctx("bấm lưu không được"))
    assert res.verify_outcome == "user_error"
    assert res.evidence["report"] is None


async def test_a_confirmed_bug_writes_the_report():
    report = BugReport(
        title="Import ký hiệu mẫu báo lỗi 500",
        summary="Import file mẫu thì báo lỗi 500.",
        steps_to_reproduce="1. Vào Lấy mẫu\n2. Bấm Import",
    )
    captured: dict = {}

    def fake(*, messages, schema, system=None, model=None):
        captured.setdefault("schemas", []).append(schema)
        return (
            _decision(outcome="bug_confirmed", reply="Đã ghi nhận.")
            if len(captured["schemas"]) == 1
            else report
        )

    with patch(
        "agent_customer_support.agents.issue_verification.complete_structured", side_effect=fake
    ):
        res = await IssueVerificationAgent().run(_ctx("đây là ảnh"))

    assert res.verify_outcome == "bug_confirmed"
    assert captured["schemas"] == [VerificationDecision, BugReport]
    assert res.evidence["report"] == report.model_dump()
    # The rest of the context survives alongside the report.
    assert res.evidence["application"] == "lay_mau_quan_trac"
    assert res.evidence["summary"] == "A bị lỗi"


async def test_a_confirmed_bug_falls_back_when_the_report_call_returns_nothing():
    with _patch(_decision(outcome="bug_confirmed", reply="Đã ghi nhận."), report=None):
        res = await IssueVerificationAgent().run(_ctx("đây là ảnh"))
    assert res.verify_outcome == "bug_confirmed"
    assert res.evidence["report"]["title"] == "A bị lỗi"
    assert res.evidence["report"]["steps_to_reproduce"] == ""


# ---- the fail-safe ----


async def test_an_unparseable_decision_asks_again_and_never_confirms():
    """A parse failure must not be able to file a ticket."""
    ctx = _ctx("gì đó")
    with patch(
        "agent_customer_support.agents.issue_verification.complete_structured", return_value=None
    ):
        res = await IssueVerificationAgent().run(ctx)
    assert res.verify_outcome == "need_more_info"
    assert res.reply
    assert res.evidence["report"] is None


async def test_an_unparseable_decision_keeps_the_slots_already_collected():
    ctx = _ctx(
        "gì đó",
        context=VerifyContext(summary="s", slots=_slots(steps="1. Bấm Import")).model_dump(),
    )
    with patch(
        "agent_customer_support.agents.issue_verification.complete_structured", return_value=None
    ):
        res = await IssueVerificationAgent().run(ctx)
    assert res.evidence["slots"]["steps"] == "1. Bấm Import"


# ---- slot filling ----


async def test_slots_are_returned_on_every_turn_so_they_survive_to_the_next():
    with _patch(_decision(steps="1. Bấm Import", actual="Báo lỗi 500")):
        res = await IssueVerificationAgent().run(_ctx("bấm import thì lỗi 500"))
    assert res.evidence["slots"]["steps"] == "1. Bấm Import"
    assert res.evidence["slots"]["actual"] == "Báo lỗi 500"


async def test_a_slot_the_model_left_empty_is_not_blanked():
    """The model sees prior turns as text and re-states what it can. An empty slot
    means "nothing new this turn", never "forget what you had"."""
    ctx = _ctx(
        "sáng nay",
        context=VerifyContext(summary="s", slots=_slots(steps="1. Bấm Import")).model_dump(),
    )
    with _patch(_decision(occurred_at="sáng nay")):
        res = await IssueVerificationAgent().run(ctx)
    assert res.evidence["slots"]["steps"] == "1. Bấm Import"
    assert res.evidence["slots"]["occurred_at"] == "sáng nay"


async def test_a_new_value_overwrites_an_old_one_so_the_user_can_correct_themselves():
    ctx = _ctx(
        "xin lỗi là lỗi 404",
        context=VerifyContext(summary="s", slots=_slots(actual="Báo lỗi 500")).model_dump(),
    )
    with _patch(_decision(actual="Báo lỗi 404")):
        res = await IssueVerificationAgent().run(ctx)
    assert res.evidence["slots"]["actual"] == "Báo lỗi 404"


async def test_a_screenshot_is_remembered_after_the_turn_it_arrived_on():
    """Prior turns' images are not re-sent to the model, so only Python can know one
    ever came."""
    att = Attachment(kind="image", media_type="image/png", data="QUJD")
    with _patch(_decision()):
        first = await IssueVerificationAgent().run(_ctx("đây là ảnh", [att]))
    assert first.evidence["has_image"] is True

    later = _ctx("còn gì nữa không", context=first.evidence)
    with _patch(_decision()):
        res = await IssueVerificationAgent().run(later)
    assert res.evidence["has_image"] is True


async def test_the_model_is_shown_what_is_already_known():
    captured: dict = {}

    def fake(*, messages, schema, system=None, model=None):
        captured["messages"] = messages
        return _decision()

    ctx = _ctx(
        "sáng nay",
        context=VerifyContext(summary="s", slots=_slots(steps="1. Bấm Import")).model_dump(),
    )
    with patch(
        "agent_customer_support.agents.issue_verification.complete_structured", side_effect=fake
    ):
        await IssueVerificationAgent().run(ctx)

    note = captured["messages"][-1]["content"]
    assert "1. Bấm Import" in note  # what we have
    assert "màn hình/menu" in note  # what we still need (a required slot)


async def test_verification_includes_prior_turns_in_messages():
    """Verification must send conversation history so the agent knows what bug
    is being verified, not just the bare follow-up message."""
    from agent_customer_support.models import Turn

    ctx = _ctx("đây là ảnh")
    ctx.conversation.turns = [
        Turn(role="user", content="thanh toán PYC không in được"),
        Turn(role="assistant", content="Bạn gửi ảnh chụp lỗi giúp mình nhé?"),
    ]
    captured: dict = {}

    def fake(*, messages, schema, system=None, model=None):
        captured["messages"] = messages
        return _decision()

    with patch(
        "agent_customer_support.agents.issue_verification.complete_structured", side_effect=fake
    ):
        await IssueVerificationAgent().run(ctx)

    # 2 prior turns + current turn + the slots note
    msgs = captured["messages"]
    assert msgs[0]["content"] == "thanh toán PYC không in được"
    assert msgs[1]["role"] == "assistant"
    current = msgs[2]["content"]
    assert current == "đây là ảnh" or (
        isinstance(current, list) and current[0].get("text") == "đây là ảnh"
    )


# ---- the code-derived report ----


async def test_fallback_report_derives_a_one_line_title():
    long = "Khi tôi bấm nút import ký hiệu mẫu " * 5 + "\nthì báo lỗi 500 và không import được"
    rep = fallback_report(long)
    assert "\n" not in rep.title
    assert len(rep.title) <= 80
    assert rep.title.startswith("Khi tôi bấm nút import")
    assert rep.summary == long
    assert rep.steps_to_reproduce == ""


async def test_fallback_report_on_empty_summary_has_a_title():
    rep = fallback_report("")
    assert rep.title


# ---- the ask-once guard (pure: no mocks) ----

from agent_customer_support.agents.issue_verification import (  # noqa: E402
    _CONFIRM_REPLY,
    _FALLBACK_SLOT_MAX,
    _SLOT_QUESTIONS,
    _guard,
)


def _verify(**kw) -> VerifyContext:
    slots = kw.pop("slots", {})
    return VerifyContext(summary="s", slots=_slots(**slots), **kw)


def test_an_answer_the_model_did_not_file_is_filed_as_written():
    v = _verify(asked_last=["module"], ask_counts={"module": 1})
    _guard(v, _decision(ask_for=["actual"]), "trên menu quy chuẩn/tiêu chuẩn")
    assert v.slots.module == "trên menu quy chuẩn/tiêu chuẩn"


def test_a_slot_the_model_did_fill_is_not_overwritten_by_the_fallback():
    v = _verify(asked_last=["module"], slots={"module": "Quy chuẩn/Tiêu chuẩn"})
    _guard(v, _decision(), "trên menu quy chuẩn/tiêu chuẩn, bấm lưu thì lỗi")
    assert v.slots.module == "Quy chuẩn/Tiêu chuẩn"


def test_the_fallback_is_trimmed():
    v = _verify(asked_last=["module"])
    _guard(v, _decision(), "x" * 500)
    assert len(v.slots.module) == _FALLBACK_SLOT_MAX


def test_an_image_only_turn_fills_nothing():
    v = _verify(asked_last=["module"])
    _guard(v, _decision(), "   ")
    assert v.slots.module == ""


def test_a_slot_already_asked_is_not_asked_again_and_the_reply_is_rebuilt():
    v = _verify(ask_counts={"module": 1}, slots={"module": ""})
    outcome, reply = _guard(
        v,
        _decision(
            reply="Cho mình xin tên màn hình và thời điểm nhé?", ask_for=["module", "occurred_at"]
        ),
        "",
    )
    assert outcome == "need_more_info"
    assert v.asked_last == ["occurred_at"]
    assert reply == _SLOT_QUESTIONS["occurred_at"]


def test_a_slot_already_filled_is_not_asked():
    v = _verify(slots={"module": "Quy chuẩn/Tiêu chuẩn"})
    _, reply = _guard(v, _decision(reply="Tên màn hình? Lỗi gì?", ask_for=["module", "actual"]), "")
    assert v.asked_last == ["actual"]
    assert reply == _SLOT_QUESTIONS["actual"]


def test_the_model_s_own_question_is_kept_when_nothing_was_filtered():
    v = _verify()
    _, reply = _guard(v, _decision(reply="Lỗi hiện thế nào ạ?", ask_for=["actual"]), "")
    assert reply == "Lỗi hiện thế nào ạ?"
    assert v.ask_counts == {"actual": 1}


def test_a_reply_that_asks_for_no_slot_is_left_alone():
    # Asking for a screenshot is not a slot; the guard must not replace it.
    v = _verify()
    _, reply = _guard(v, _decision(reply="Bạn gửi giúp mình ảnh lỗi nhé?"), "")
    assert reply == "Bạn gửi giúp mình ảnh lỗi nhé?"


def test_every_required_slot_filled_confirms_the_bug():
    v = _verify(slots={"module": "Quy chuẩn/Tiêu chuẩn", "actual": "Báo lỗi 500"})
    outcome, reply = _guard(v, _decision(reply="Thêm bước?", ask_for=["steps"]), "")
    assert (outcome, reply) == ("bug_confirmed", _CONFIRM_REPLY)


def test_when_every_missing_required_slot_was_asked_the_bug_is_confirmed_anyway():
    v = _verify(ask_counts={"module": 1, "actual": 1})
    outcome, _ = _guard(v, _decision(ask_for=["module"]), "")
    assert outcome == "bug_confirmed"


def test_when_the_model_s_asks_are_all_filtered_a_missing_required_slot_is_asked_once():
    v = _verify(slots={"module": "m"}, ask_counts={"steps": 1})
    outcome, reply = _guard(v, _decision(ask_for=["steps"]), "")
    assert outcome == "need_more_info"
    assert reply == _SLOT_QUESTIONS["actual"]
    assert v.asked_last == ["actual"]


def test_a_user_error_passes_through_the_guard():
    v = _verify(asked_last=["module"])
    outcome, reply = _guard(v, _decision(outcome="user_error", reply="Cần chọn mẫu trước."), "à ok")
    assert (outcome, reply) == ("user_error", "Cần chọn mẫu trước.")
    assert v.asked_last == []


# ---- the guard inside a real run ----


async def test_the_conversation_from_the_bug_report_stops_asking_for_the_screen():
    """The live bug: the model kept leaving `module` empty and asking for it, three
    turns running, after the user had named it twice."""
    stuck = _decision(
        reply="Bạn cho mình xin đúng tên màn hình/menu cụ thể nhé?", ask_for=["module"]
    )

    ctx = _ctx("đây là màn hình báo lỗi, tôi ko thể nhập chỉ tiêu cho quy chuẩn")
    with _patch(stuck):
        turn1 = await IssueVerificationAgent().run(ctx)
    assert turn1.evidence["asked_last"] == ["module"]

    ctx = _ctx(
        "hệ thống báo lỗi như màn hình, trên menu quy chuẩn/tiêu chuẩn", context=turn1.evidence
    )
    with _patch(stuck):
        turn2 = await IssueVerificationAgent().run(ctx)
    assert "quy chuẩn/tiêu chuẩn" in turn2.evidence["slots"]["module"]
    assert "màn hình/menu" not in turn2.reply


async def test_a_promoted_confirmation_writes_the_report():
    report = BugReport(title="Không nhập được chỉ tiêu", summary="s", steps_to_reproduce="")
    ctx = _ctx(
        "lỗi như ảnh",
        context=VerifyContext(
            summary="s", slots=_slots(module="Quy chuẩn/Tiêu chuẩn", actual="Báo lỗi 500")
        ).model_dump(),
    )
    with _patch(_decision(ask_for=["steps"]), report=report):
        res = await IssueVerificationAgent().run(ctx)
    assert res.verify_outcome == "bug_confirmed"
    assert res.evidence["report"]["title"] == "Không nhập được chỉ tiêu"


async def test_a_parse_failure_still_files_the_answer_to_last_turn_s_question():
    ctx = _ctx(
        "menu quy chuẩn/tiêu chuẩn",
        context=VerifyContext(summary="s", asked_last=["module"]).model_dump(),
    )
    with patch(
        "agent_customer_support.agents.issue_verification.complete_structured", return_value=None
    ):
        res = await IssueVerificationAgent().run(ctx)
    assert res.verify_outcome == "need_more_info"
    assert res.evidence["slots"]["module"] == "menu quy chuẩn/tiêu chuẩn"


async def test_earlier_screenshots_are_shown_to_the_model_again():
    captured: dict = {}

    def fake(*, messages, schema, system=None, model=None):
        captured["messages"] = messages
        return _decision()

    ctx = _ctx("trên menu quy chuẩn/tiêu chuẩn")
    ctx.evidence_images = [Attachment(kind="image", media_type="image/png", data="QUJD")]
    with patch(
        "agent_customer_support.agents.issue_verification.complete_structured", side_effect=fake
    ):
        res = await IssueVerificationAgent().run(ctx)

    note = captured["messages"][-1]["content"]
    assert isinstance(note, list)  # multimodal: text + the earlier screenshot
    assert any("QUJD" in str(block) for block in note)
    # The user's own message this turn is left exactly as written.
    assert captured["messages"][-2]["content"] == "trên menu quy chuẩn/tiêu chuẩn"
    assert res.evidence["has_image"] is True

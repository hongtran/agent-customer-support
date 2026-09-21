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


def _decision(outcome="need_more_info", reply="...", **slots) -> VerificationDecision:
    return VerificationDecision(outcome=outcome, reply=reply, slots=_slots(**slots))


def _ctx(message, attachments=None, context=None) -> TurnContext:
    s = SessionState(
        conversation_id="cv1",
        pending="verify_issue",
        pending_context=(
            context
            if context is not None
            else VerifyContext(application="lay_mau_quan_trac", summary="A bị lỗi").model_dump()
        ),
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
    assert "kết quả mong đợi" in note  # what we still need


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

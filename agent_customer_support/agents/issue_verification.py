import logging
from collections.abc import Sequence
from typing import Literal

from qdrant_client.http.exceptions import ApiException

from agent_customer_support import doc_images
from agent_customer_support.agents.context import TurnContext
from agent_customer_support.agents.passages import passages_block
from agent_customer_support.agents.prompts import (
    BUG_REPORT_PROMPT,
    ISSUE_DOC_CHECK_PROMPT,
    ISSUE_VERIFICATION_PROMPT,
    PROCESS_BLOCK,
    VERIFICATION_SLOTS_NOTE,
)
from agent_customer_support.citations import PROCESS_DOC_ID
from agent_customer_support.config import get_settings
from agent_customer_support.llm import complete_structured
from agent_customer_support.llm.normalize import (
    to_anthropic_content,
    to_openai_content,
)
from agent_customer_support.llm.schemas import BugReport, DocCheck, VerificationDecision
from agent_customer_support.models import AgentResult, VerifyContext
from agent_customer_support.observability import tracing

logger = logging.getLogger(__name__)

_FALLBACK_TITLE_MAX = 80
_FALLBACK_TITLE = "Lỗi do người dùng báo qua chatbot"

# Used when the structured call returns nothing. Deliberately a re-ask and never a
# confirmation: a parse failure must not be able to file a ticket.
_FALLBACK_REPLY = (
    "Mình chưa ghi nhận được đầy đủ thông tin. Bạn mô tả giúp mình các bước đã thao "
    "tác và điều bạn thấy trên màn hình nhé (hoặc gửi ảnh chụp màn hình)."
)

# How much of the user's answer lands in a slot when the model did not fill it.
_FALLBACK_SLOT_MAX = 200

# Used when the guard has to write the reply itself — a slot was filtered out of what
# the model asked, so its own text would ask something we must not ask again.
_SLOT_QUESTIONS: dict[str, str] = {
    "module": "Bạn đang thao tác ở menu/màn hình nào khi gặp lỗi?",
    "version": "Bạn đang dùng phần mềm trên web hay bản cài trên máy (desktop)?",
    "steps": "Bạn mô tả giúp mình các bước đã làm trước khi gặp lỗi nhé?",
    "expected": "Bạn mong hệ thống thực hiện hoặc hiển thị điều gì?",
    "actual": "Hệ thống báo lỗi hoặc hiển thị như thế nào (nguyên văn thông báo nếu có)?",
    "occurred_at": "Lỗi xảy ra lúc nào, và có lặp lại không?",
}

_CONFIRM_REPLY = "Mình đã ghi nhận đủ thông tin lỗi và sẽ chuyển cho đội kỹ thuật kiểm tra."

_PRIOR_IMAGES_NOTE = "Ảnh chụp màn hình người dùng đã gửi ở các lượt trước:"

_DOC_CHECK_NOTE = (
    "Đoạn trích hướng dẫn sử dụng:\n{passages}\n\n"
    "Hãy so sánh điều người dùng mô tả ở trên với QUY TRÌNH và các đoạn trích này."
)


def fallback_report(summary: str) -> BugReport:
    """Code-derived report for when the structured call returns nothing.

    The first line of the user's original message, cut at a word boundary, is a
    worse title than the model's but a far better one than an empty ticket — and
    the ticket must still be filed, because the evidence has already been collected.
    """
    first_line = next((ln.strip() for ln in summary.splitlines() if ln.strip()), "")
    title = first_line
    if len(title) > _FALLBACK_TITLE_MAX:
        cut = title[:_FALLBACK_TITLE_MAX].rsplit(" ", 1)[0]
        title = (cut or title[:_FALLBACK_TITLE_MAX]).rstrip() + "…"
    return BugReport(title=title or _FALLBACK_TITLE, summary=summary, steps_to_reproduce="")


class IssueVerificationAgent:
    name = "issue_verification"

    async def run(self, ctx: TurnContext) -> AgentResult:
        """Collect one turn's worth of evidence and say what it amounts to.

        Slot filling: the agent reports the whole fixed set of facts a ticket needs
        every turn, and Python merges that reading into what the session already
        held. The merge matters because the model is not the memory here — it sees
        prior turns as plain text, so a slot it leaves empty means "nothing new",
        never "forget that". Earlier screenshots are re-sent (`ctx.evidence_images`,
        loaded by the coordinator) so a screen name or an error dialog in them can
        still be read on a later turn.

        `_guard` then makes sure the collection moves forward: a slot is asked at most
        once, and an answer the model failed to file is filed as written.

        The caller owns when to stop. This method never decides that the collection
        has gone on too long; `routing.next_step` does, from `verify_turns`.
        """
        cfg = get_settings()
        model = cfg.model_for("issue_verification")
        is_anthropic = "claude" in model
        to_content = to_anthropic_content if is_anthropic else to_openai_content
        # Prior turns as plain text so the agent knows what bug is being verified.
        history = [{"role": t.role, "content": t.content} for t in ctx.conversation.turns]
        # Current turn carries multimodal content (the user may attach a screenshot).
        messages = history + [{"role": "user", "content": to_content(ctx.message, ctx.attachments)}]

        verify = VerifyContext.model_validate(ctx.session.pending_context or {})
        # A screenshot that arrived on an earlier turn is not in the history — only its
        # text is — so this flag is the only record that one was ever sent.
        verify.has_image = (
            verify.has_image
            or bool(ctx.evidence_images)
            or any(a.kind == "image" for a in ctx.attachments)
        )

        if not verify.doc_checked:
            # Once per bug, before any slot is asked: if the guides say the software
            # behaves exactly like this, there is nothing to collect. Marked first, so a
            # failed check is not paid for again on the next turn.
            verify.doc_checked = True
            check = await _doc_check(ctx, verify, messages, model)
            if check is not None:
                verify.doc_expected = check.doc_expected.strip()
                if check.verdict == "works_as_documented":
                    return AgentResult(
                        reply=check.explanation.strip(),
                        verify_outcome="user_error",
                        evidence=verify.model_dump(),
                    )
        logger.warning(verify.model_dump_json(indent=2, ensure_ascii=False))
        logger.warning(f"Evidence images: {ctx.evidence_images}")
        decision = complete_structured(
            messages=messages
            + [{"role": "user", "content": _note_content(verify, ctx, to_content)}],
            schema=VerificationDecision,
            system=ISSUE_VERIFICATION_PROMPT,
            model=model,
        )
        if decision is None:
            # Fail safe, never forward: keep what we had, file this turn's answer if it
            # was one, and ask again. The turn still counts against the cap.
            _fill_answered(verify, ctx.message)
            verify.asked_last = []
            return AgentResult(
                reply=_FALLBACK_REPLY,
                verify_outcome="need_more_info",
                evidence=verify.model_dump(),
            )

        verify.slots = verify.slots.merge(decision.slots)
        outcome, reply = _guard(verify, decision, ctx.message)

        if outcome == "bug_confirmed":
            # One structured call over the same conversation turns it into the ticket
            # text. Labelled so it lands as `llm.issue_verification.report` in the trace.
            with tracing.step("report"):
                report = complete_structured(
                    messages=messages,
                    schema=BugReport,
                    system=BUG_REPORT_PROMPT,
                    model=model,
                )
            if report is None:
                report = fallback_report(verify.summary or ctx.message)
            verify.report = report.model_dump()

        return AgentResult(reply=reply, verify_outcome=outcome, evidence=verify.model_dump())


def _fill_answered(verify: VerifyContext, message: str) -> None:
    """File the user's message into every slot we asked for last turn that is still empty.

    The previous reply asked for these slots, so this message is the answer — and if
    the model still left one empty, it failed to read the answer, not the user failed
    to give it. Filing the words as written is worse than a clean name and far better
    than asking a third time. An image-only turn has no words, and fills nothing.

    Known limit: this assumes the message answers the question. An off-topic reply
    lands in the slot. A cancel phrase never reaches here — `routing.wants_cancel`
    drops the flow first.
    """
    text = (message or "").strip()
    if not text:
        return
    updates = {
        slot: text[:_FALLBACK_SLOT_MAX]
        for slot in verify.asked_last
        if slot in _SLOT_QUESTIONS and not getattr(verify.slots, slot).strip()
    }
    if updates:
        verify.slots = verify.slots.model_copy(update=updates)


def _questions(slots: Sequence[str]) -> str:
    return " ".join(_SLOT_QUESTIONS[s] for s in slots)


Outcome = Literal["need_more_info", "user_error", "bug_confirmed"]


def _guard(
    verify: VerifyContext, decision: VerificationDecision, message: str
) -> tuple[Outcome, str]:
    """Make sure the collection moves forward. Returns (outcome, reply).

    Pure apart from updating `verify`, so each rule is a plain unit test. Code does
    this rather than the prompt because the prompt alone did not hold: the model asked
    for the same screen name three turns running after the user had given it twice.

    Rules, in order:
      1. an answer to last turn's question that the model did not file is filed as
         written (`_fill_answered`);
      2. a slot already filled, or already asked once, is never asked again;
      3. while still collecting — every required slot filled means there is nothing
         left to wait for, so the bug is confirmed; if everything the model asked for
         was filtered out but a required slot is still empty, one canned question for
         it, or, if every one of those was already asked, the bug is confirmed with
         what we have (the ticket body names what is missing). A reply that asks for
         no slot at all — a screenshot, say — is left alone;
      4. if a slot was filtered out of what the model asked, its reply would ask it
         anyway, so the reply is rebuilt from the canned questions.

    `user_error` and the model's own `bug_confirmed` pass through untouched.
    """
    _fill_answered(verify, message)
    outcome: Outcome = decision.outcome
    reply = decision.reply
    if outcome != "need_more_info":
        verify.asked_last = []
        return outcome, reply

    ask: list[str] = [
        s
        for s in decision.ask_for
        if not getattr(verify.slots, s).strip() and verify.ask_counts.get(s, 0) == 0
    ]
    missing = verify.slots.missing_slots()
    if not missing:
        outcome, reply, ask = "bug_confirmed", _CONFIRM_REPLY, []
    elif decision.ask_for and not ask:
        unasked = [s for s in missing if verify.ask_counts.get(s, 0) == 0]
        if unasked:
            ask = unasked[:1]
            reply = _questions(ask)
        else:
            outcome, reply = "bug_confirmed", _CONFIRM_REPLY
    elif len(ask) < len(decision.ask_for):
        reply = _questions(ask)

    verify.asked_last = list(ask)
    for s in ask:
        verify.ask_counts[s] = verify.ask_counts.get(s, 0) + 1
    return outcome, reply


def _note_content(verify: VerifyContext, ctx: TurnContext, to_content):
    """Tell the model what is already known and asked, and show earlier screenshots.

    Appended as user content rather than folded into the system prompt: the system
    text is the same on every turn and stays cacheable, while this block changes
    every turn by definition. Earlier screenshots ride here rather than on the current
    message, so the user's own words this turn stay exactly as they wrote them.
    """
    filled = verify.slots.describe()
    if verify.has_image:
        filled = f"{filled}\n- ảnh chụp màn hình: đã có" if filled else "- ảnh chụp màn hình: đã có"
    asked = [s for s, n in verify.ask_counts.items() if n > 0]
    note = VERIFICATION_SLOTS_NOTE.format(
        filled=filled or "(chưa có gì)",
        doc_expected=verify.doc_expected or "(tài liệu không nói về trường hợp này)",
        missing=", ".join(verify.slots.missing()) or "(không thiếu gì bắt buộc)",
        asked=", ".join(asked) or "(chưa hỏi gì)",
    )
    if not ctx.evidence_images:
        return note
    return to_content(f"{note}\n\n{_PRIOR_IMAGES_NOTE}", ctx.evidence_images)


async def _doc_check(
    ctx: TurnContext, verify: VerifyContext, messages: list[dict], model: str
) -> DocCheck | None:
    """Compare the report with what the guides say, before collecting anything.

    Same sources as the knowledge composer: PROCESS_BLOCK in the system prefix (so the
    cached prefix is shared) and the guide passages, searched with the same scope rules
    (`search_with_fallback`, never wider than what the customer bought). The model is
    called even with no passages, because the process block alone can show the
    behavior is correct.

    Returns None when the call produced nothing. A `works_as_documented` verdict that
    cites no id we actually offered is downgraded to `not_covered`: the catalog, not the
    shape, is the guard, and closing a real bug as a user error is the costly mistake.
    """
    passages = await _search_guides(ctx, verify)
    note = _DOC_CHECK_NOTE.format(passages=passages_block(passages, with_sections=True) or "(rỗng)")
    with tracing.step("doc_check"):
        check = complete_structured(
            messages=messages + [{"role": "user", "content": note}],
            schema=DocCheck,
            system=[PROCESS_BLOCK, {"type": "text", "text": ISSUE_DOC_CHECK_PROMPT}],
            model=model,
        )
    if check is None:
        logger.warning("doc check returned nothing, collecting evidence instead")
        return None
    if check.verdict == "works_as_documented":
        offered = {str(i) for i in range(len(passages))} | {PROCESS_DOC_ID}
        if not any(c.strip() in offered for c in check.cited) or not check.explanation.strip():
            return check.model_copy(update={"verdict": "not_covered"})
    return check


async def _search_guides(ctx: TurnContext, verify: VerifyContext) -> list[str]:
    """The guide passages for this report, text only.

    Knowledge's contextualized query when the bug came from there, else the user's own
    words. Image refs are dropped (an empty catalog) — this call only reads text, and a
    leftover `media/…` ref would only be noise. A Qdrant failure degrades to the process
    block alone rather than failing the turn.
    """
    if ctx.rag is None:
        return []
    cfg = get_settings()
    query = verify.query or verify.summary or ctx.message
    try:
        res = await ctx.rag.search_with_fallback(
            query,
            collection=cfg.product_collection,
            applications=ctx.session.selected_applications or None,
            fallback_applications=ctx.customer.enabled_applications or None,
        )
    except (ApiException, ValueError) as exc:
        logger.warning("doc check search failed, using process only: %s", exc)
        return []
    passages = res.get("passages", []) or []
    return doc_images.rewrite_passages(passages, res.get("metas", []) or [], {})

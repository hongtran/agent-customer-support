from agent_customer_support.agents.context import TurnContext
from agent_customer_support.agents.prompts import (
    BUG_REPORT_PROMPT,
    ISSUE_VERIFICATION_PROMPT,
    VERIFICATION_SLOTS_NOTE,
)
from agent_customer_support.config import get_settings
from agent_customer_support.llm import complete_structured
from agent_customer_support.llm.normalize import (
    to_anthropic_content,
    to_openai_content,
)
from agent_customer_support.llm.schemas import BugReport, VerificationDecision
from agent_customer_support.models import AgentResult, BugSlots, VerifyContext
from agent_customer_support.observability import tracing

_FALLBACK_TITLE_MAX = 80
_FALLBACK_TITLE = "Lỗi do người dùng báo qua chatbot"

# Used when the structured call returns nothing. Deliberately a re-ask and never a
# confirmation: a parse failure must not be able to file a ticket.
_FALLBACK_REPLY = (
    "Mình chưa ghi nhận được đầy đủ thông tin. Bạn mô tả giúp mình các bước đã thao "
    "tác và điều bạn thấy trên màn hình nhé (hoặc gửi ảnh chụp màn hình)."
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
        prior turns as plain text and prior screenshots not at all, so a slot it
        leaves empty means "nothing new", never "forget that".

        The caller owns when to stop. This method never decides that the collection
        has gone on too long; `routing.next_step` does, from `verify_turns`.
        """
        cfg = get_settings()
        # The settings field is still `verification_model` (env `VERIFICATION_MODEL`),
        # so the lookup key stays "verification" — renaming it would silently drop the
        # override back to `agent_model` on every existing deployment.
        model = cfg.model_for("issue_verification")
        is_anthropic = "claude" in model
        # Prior turns as plain text so the agent knows what bug is being verified.
        history = [{"role": t.role, "content": t.content} for t in ctx.conversation.turns]
        # Current turn carries multimodal content (the user may attach a screenshot).
        if is_anthropic:
            current_content = to_anthropic_content(ctx.message, ctx.attachments)
        else:
            current_content = to_openai_content(ctx.message, ctx.attachments)
        messages = history + [{"role": "user", "content": current_content}]

        verify = VerifyContext.model_validate(ctx.session.pending_context or {})
        # A screenshot that arrived on an earlier turn is not in `messages` — only its
        # text is — so this flag is the only record that one was ever sent.
        verify.has_image = verify.has_image or any(a.kind == "image" for a in ctx.attachments)

        decision = complete_structured(
            messages=messages + [{"role": "user", "content": _slots_note(verify)}],
            schema=VerificationDecision,
            system=ISSUE_VERIFICATION_PROMPT,
            model=model,
        )
        if decision is None:
            # Fail safe, never forward: keep the slots we had and ask again. The turn
            # still counts against the cap, so this cannot loop.
            verify.slots = verify.slots.merge(BugSlots.empty())
            return AgentResult(
                reply=_FALLBACK_REPLY,
                verify_outcome="need_more_info",
                evidence=verify.model_dump(),
            )

        verify.slots = verify.slots.merge(decision.slots)

        if decision.outcome == "bug_confirmed":
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

        return AgentResult(
            reply=decision.reply,
            verify_outcome=decision.outcome,
            evidence=verify.model_dump(),
        )


def _slots_note(verify: VerifyContext) -> str:
    """Tell the model what is already known, so it asks only for what is not.

    Appended as user content rather than folded into the system prompt: the system
    text is the same on every turn and stays cacheable, while this block changes
    every turn by definition.
    """
    filled = verify.slots.describe()
    missing = verify.slots.missing()
    if verify.has_image:
        filled = f"{filled}\n- ảnh chụp màn hình: đã có" if filled else "- ảnh chụp màn hình: đã có"
    return VERIFICATION_SLOTS_NOTE.format(
        filled=filled or "(chưa có gì)",
        missing=", ".join(missing) or "(không thiếu gì bắt buộc)",
    )

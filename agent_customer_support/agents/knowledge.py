import re

from agent_customer_support.agents.context import TurnContext
from agent_customer_support.agents.prompts import (
    KNOWLEDGE_CONTEXTUALIZE_PROMPT,
    KNOWLEDGE_CONTEXTUALIZE_VISION_PROMPT,
    KNOWLEDGE_COMPOSE_PROMPT,
    KNOWLEDGE_RESUME_NO_CLARIFY,
    PROCESS_BLOCK,
)
from agent_customer_support.config import Settings, get_settings
from agent_customer_support.llm import complete_text
from agent_customer_support.llm.normalize import (
    to_anthropic_content,
    to_openai_content,
)
from agent_customer_support.models import AgentResult, QARecord

_NO_ANSWER_RE = re.compile(r"\[\[no_answer\]\]")
_BUG_RE = re.compile(r"\[\[suspected_bug:([a-zA-Z0-9_\-]+)\]\]")
_CLARIFY_RE = re.compile(r"\[\[clarify\]\]")


def _scrub_markers(text: str) -> str:
    """Strip every known marker pattern from text, regardless of kind.

    Guards against the model emitting a stray second marker (e.g. both
    [[no_answer]] and [[clarify]]): the selected kind drives routing, but no
    marker should ever leak literally into the user-facing reply.
    """
    for pattern in (_BUG_RE, _CLARIFY_RE, _NO_ANSWER_RE):
        text = pattern.sub("", text)
    return text.strip()


def parse_markers(text: str) -> tuple[str, str | None, str | None]:
    """Return (clean_text, kind, application) where kind in
    {None, 'no_answer', 'suspected_bug', 'clarify'}.

    Precedence: suspected_bug > clarify > no_answer. A bug is the safest handoff,
    so it wins if the model emits more than one marker.
    """
    bug = _BUG_RE.search(text or "")
    if bug:
        return _scrub_markers(text or ""), "suspected_bug", bug.group(1)
    if _CLARIFY_RE.search(text or ""):
        return _scrub_markers(text or ""), "clarify", None
    if _NO_ANSWER_RE.search(text or ""):
        clean = _scrub_markers(text or "")
        # If the model wrote substantial content AND appended [[no_answer]], the marker
        # is a spurious hedge — trust the content and treat it as a valid answer.
        if len(clean) > 80:
            return clean, None, None
        return clean, "no_answer", None
    return (text or "").strip(), None, None


def _passages_block(passages: list[str]) -> str:
    return "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages))


_HAS_PRIOR_TURN = "assistant:"


class KnowledgeAgent:
    name = "knowledge"

    async def _contextualize(self, ctx: TurnContext, cfg: Settings) -> str:
        """Resolve pronouns/references in ctx.message into a standalone search query.

        Two sources of context are folded in:
          - conversation history (resolves "xoá nó thì sao?" → "cách xoá mẫu xét nghiệm")
          - an attached screenshot (resolves "cái này lỗi gì?" → names the page/feature
            visible in the image), so a user who can't name the feature can just point

        Returns ctx.message unchanged only when there is nothing to resolve — i.e. the
        first turn AND no screenshot — to avoid a pointless LLM call.
        """
        has_image = any(a.kind == "image" for a in ctx.attachments)
        if _HAS_PRIOR_TURN not in ctx.transcript and not has_image:
            return ctx.message

        model = cfg.model_for("knowledge_contextualize")
        if has_image:
            text = f"{ctx.transcript}\n\n(Ảnh chụp màn hình người dùng đính kèm bên dưới.)"
            if "claude" in model:
                content: object = to_anthropic_content(text, ctx.attachments)
            else:
                content = to_openai_content(text, ctx.attachments)
            system = KNOWLEDGE_CONTEXTUALIZE_VISION_PROMPT
        else:
            content = ctx.transcript
            system = KNOWLEDGE_CONTEXTUALIZE_PROMPT

        raw = complete_text(
            messages=[{"role": "user", "content": content}],
            system=system,
            model=model,
        )
        return (raw or ctx.message).strip()

    async def _compose(
        self,
        question: str,
        passages: list[str],
        transcript: str,
        cfg: Settings,
        allow_clarify: bool = True,
    ) -> str:
        """Compose a grounded answer from the always-on process + retrieved passages.

        The process context is injected as a cached system prefix (PROCESS_BLOCK);
        passages carry the per-module detail and may be empty. Transcript is passed
        for context so the model can resolve remaining ambiguity in phrasing — but
        content must still come only from the two sources (enforced by the prompt).
        """
        if _HAS_PRIOR_TURN in transcript:
            history = f"Lịch sử hội thoại:\n{transcript}\n\n"
        else:
            history = ""
        content = (
            f"{history}Câu hỏi hiện tại: {question}\n\nĐoạn trích:\n{_passages_block(passages)}"
        )
        if not allow_clarify:
            content = f"{content}\n\n{KNOWLEDGE_RESUME_NO_CLARIFY}"
        return complete_text(
            messages=[{"role": "user", "content": content}],
            system=[PROCESS_BLOCK, {"type": "text", "text": KNOWLEDGE_COMPOSE_PROMPT}],
            model=cfg.model_for("knowledge"),
        )

    async def run(self, ctx: TurnContext) -> AgentResult:
        """
        Single-attempt pipeline:
          contextualize → search → compose (process always-on) → return result

        On a miss (compose emits [[no_answer]] — neither process nor passages answer):
          - first miss → ask ONE clarifying question (and invite a screenshot), so the
            user can pin down a vague or jargon-y request; state kept in
            session.pending = "knowledge_clarify" to bound this to a single attempt
          - second miss (the clarification turn) → log to backlog and hand off
        suspected_bug returns immediately without backlog.
        """
        cfg = get_settings()
        # If we asked a clarifying question last turn, this turn is the answer to it.
        # Consume the flag now so we never clarify twice in a row.
        already_clarified = ctx.session.pending == "knowledge_clarify"
        if already_clarified:
            ctx.session.pending = None
        query = await self._contextualize(ctx, cfg)

        applications = ctx.session.selected_applications or None
        res = await ctx.rag.search(
            query, collection=cfg.product_collection, applications=applications
        )
        passages = res.get("passages", []) or []
        citations = res.get("citations", []) or []

        # Always compose: the process context is always in the system prefix, so even
        # with no retrieved passages the model can answer process-level questions.
        # The [[no_answer]] marker is the single miss signal — emitted only when
        # neither the process nor the passages can answer.
        composed = await self._compose(
            query, passages, ctx.transcript, cfg, allow_clarify=not already_clarified
        )
        clean, kind, application = parse_markers(composed)

        if kind == "suspected_bug":
            return AgentResult(
                reply=clean,
                resolved=False,
                suspected_bug=True,
                evidence={"application": application, "summary": ctx.message},
                citations=citations,
            )

        # Clarify / confirm before answering. The composer judged that an element it
        # can't see (ambiguous subject, unknown user-state, unverified premise, or a
        # risky intent) materially changes the answer. Ask once — bounded by the same
        # knowledge_clarify flag — then re-ground on the user's reply next turn.
        # allow_clarify=False on the resume turn means compose should never reach here
        # twice; if the model disobeys, downgrade to a plain (assumption-stated) answer.
        if kind == "clarify":
            if not already_clarified:
                ctx.session.pending = "knowledge_clarify"
                return AgentResult(reply=clean, resolved=None, citations=citations)
            return AgentResult(reply=clean, resolved=True, citations=citations)

        if kind != "no_answer":
            return AgentResult(reply=clean, resolved=True, citations=citations)

        # Miss. On the first one, try to disambiguate before giving up to a human:
        # the request may just be vague or use the customer's own terminology.
        if not already_clarified:
            ctx.session.pending = "knowledge_clarify"
            return AgentResult(
                reply="Mình chưa rõ ý bạn lắm. Bạn cho mình biết cụ thể hơn đang thao tác "
                "ở màn hình/chức năng nào nhé — hoặc chụp giúp mình ảnh màn hình "
                "đang xem để mình hỗ trợ nhanh hơn.",
                resolved=None,
                citations=citations,
            )

        # Second miss after a clarification: genuinely not in the KB — log and hand off.
        await ctx.backlog.add(
            customer_id=ctx.customer.customer_id,
            type="how_to_missing",
            summary=ctx.message,
            application=None,
            transcript=ctx.transcript,
        )
        await ctx.qa_store.add(
            QARecord(
                question=ctx.message,
                source="cannot_answer",
                customer_id=ctx.customer.customer_id,
                conversation_id=ctx.session.conversation_id,
                transcript=ctx.transcript,
            )
        )
        return AgentResult(
            reply="Mình chưa tìm thấy thông tin cụ thể này trong tài liệu. "
            "Đã ghi nhận để đội hỗ trợ bổ sung.",
            resolved=False,
            citations=citations,
        )

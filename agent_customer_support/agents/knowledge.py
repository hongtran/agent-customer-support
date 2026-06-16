import json
import re

from agent_customer_support.agents.context import TurnContext
from agent_customer_support.agents.prompts import (
    KNOWLEDGE_CONTEXTUALIZE_PROMPT,
    KNOWLEDGE_CONTEXTUALIZE_VISION_PROMPT,
    KNOWLEDGE_GRADER_PROMPT,
    KNOWLEDGE_COMPOSE_PROMPT,
    DIAGNOSTIC_PROMPT,
)
from agent_customer_support.agents.diagnostics import (
    DIAGNOSTIC_RULES,
    RULES_BY_ID,
    DiagnosticRule,
)
from agent_customer_support.config import Settings, get_settings
from agent_customer_support.llm import complete_text
from agent_customer_support.llm.normalize import (
    to_anthropic_content,
    to_openai_content,
)
from agent_customer_support.models import AgentResult

_NO_ANSWER_RE = re.compile(r"\[\[no_answer\]\]")
_BUG_RE = re.compile(r"\[\[suspected_bug:([a-zA-Z0-9_\-]+)\]\]")

HIGH = 0.70
LOW = 0.50
MIN_SUBSTANTIAL_CHARS = 200  # high score but shorter than this => grade


def needs_grading(top_confidence: float, passages: list[str]) -> bool:
    """Return True if the LLM grader should be consulted to confirm answer presence."""
    if not passages:
        return False
    if LOW <= top_confidence < HIGH:
        return True
    if top_confidence >= HIGH:
        total = sum(len(p) for p in passages)
        return total < MIN_SUBSTANTIAL_CHARS
    return False  # low score: skip grader, reformulate instead


def parse_markers(text: str) -> tuple[str, str | None, str | None]:
    """Return (clean_text, kind, module) where kind in {None, 'no_answer', 'suspected_bug'}."""
    bug = _BUG_RE.search(text or "")
    if bug:
        clean = _BUG_RE.sub("", text).strip()
        return clean, "suspected_bug", bug.group(1)
    if _NO_ANSWER_RE.search(text or ""):
        clean = _NO_ANSWER_RE.sub("", text).strip()
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

    async def _grade(self, question: str, passages: list[str], cfg: Settings) -> bool:
        """Ask the LLM grader whether the passages actually answer the question."""
        raw = complete_text(
            messages=[
                {
                    "role": "user",
                    "content": f"CÂU HỎI: {question}\n\nĐOẠN TRÍCH:\n{_passages_block(passages)}",
                }
            ],
            system=KNOWLEDGE_GRADER_PROMPT,
            model=cfg.model_for("knowledge_grader"),
        )
        try:
            return bool(json.loads(raw).get("answer_present"))
        except (json.JSONDecodeError, TypeError):
            return False  # fail-closed: don't answer if grader is unparseable

    async def _diagnose(self, query: str, cfg: Settings) -> DiagnosticRule | None:
        """Classify the query against known operating-principle symptoms.

        Returns the matched DiagnosticRule, or None when nothing matches or the
        classifier output is unusable. Fail-closed by design: a diagnostic failure
        must never block an answer the pipeline would otherwise produce.
        """
        rules_block = "\n".join(f"{r.id}: {r.symptom}" for r in DIAGNOSTIC_RULES)
        raw = complete_text(
            messages=[
                {
                    "role": "user",
                    "content": f"DANH SÁCH QUY TẮC:\n{rules_block}\n\nCÂU HỎI: {query}",
                }
            ],
            system=DIAGNOSTIC_PROMPT,
            model=cfg.model_for("diagnostic"),
        )
        try:
            rule_id = json.loads(raw).get("rule_id")
        except (json.JSONDecodeError, TypeError, AttributeError):
            return None
        if not rule_id or rule_id == "none":
            return None
        return RULES_BY_ID.get(rule_id)  # unknown id -> None

    async def _compose(
        self, question: str, passages: list[str], transcript: str, cfg: Settings
    ) -> str:
        """Compose a grounded answer from passages, with optional conversation history.

        Passes the transcript as context so the model can resolve remaining ambiguity
        in phrasing and avoid repeating information already given — but content must
        still come only from passages (enforced by KNOWLEDGE_COMPOSE_PROMPT).
        """
        if _HAS_PRIOR_TURN in transcript:
            history = f"Lịch sử hội thoại:\n{transcript}\n\n"
        else:
            history = ""
        content = (
            f"{history}Câu hỏi hiện tại: {question}\n\nĐoạn trích:\n{_passages_block(passages)}"
        )
        return complete_text(
            messages=[{"role": "user", "content": content}],
            system=KNOWLEDGE_COMPOSE_PROMPT,
            model=cfg.model_for("knowledge"),
        )

    async def _present(
        self, question: str, passages: list[str], conf: float, cfg: Settings
    ) -> bool:
        """Determine whether the passages are present/relevant enough to attempt composing."""
        # return True
        if not passages:
            return False
        if needs_grading(conf, passages):
            return await self._grade(question, passages, cfg)
        return conf >= HIGH  # trust high, distrust low

    async def run(self, ctx: TurnContext) -> AgentResult:
        """
        Single-attempt pipeline:
          contextualize → search → check presence → compose → return result

        On a miss (no relevant passages, or compose emits [[no_answer]]):
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

        modules = ctx.session.selected_modules or None
        res = await ctx.rag.search(query, collection=cfg.product_collection, modules=modules)
        passages = res.get("passages", []) or []
        conf = res.get("top_confidence", 0.0)
        citations = res.get("citations", []) or []

        present = await self._present(query, passages, conf, cfg)
        if present:
            composed = await self._compose(query, passages, ctx.transcript, cfg)
            clean, kind, module = parse_markers(composed)

            if kind == "suspected_bug":
                return AgentResult(
                    reply=clean,
                    resolved=False,
                    suspected_bug=True,
                    evidence={"module": module, "summary": ctx.message},
                    citations=citations,
                )

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
            module=None,
            transcript=ctx.transcript,
        )
        return AgentResult(
            reply="Mình chưa tìm thấy thông tin cụ thể này trong tài liệu. "
            "Đã ghi nhận để đội hỗ trợ bổ sung.",
            resolved=False,
            citations=citations,
        )

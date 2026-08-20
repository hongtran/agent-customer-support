import logging
import re

from qdrant_client.http.exceptions import ApiException

from agent_customer_support.agents.context import TurnContext
from agent_customer_support.agents.prompts import (
    KNOWLEDGE_CONTEXTUALIZE_PROMPT,
    KNOWLEDGE_CONTEXTUALIZE_VISION_PROMPT,
    KNOWLEDGE_COMPOSE_PROMPT,
    KNOWLEDGE_COMPOSE_PROMPT_WITH_QA,
    KNOWLEDGE_OTHER_APPLICATION_NOTE,
    KNOWLEDGE_RESUME_NO_CLARIFY,
    PROCESS_BLOCK,
)
from agent_customer_support import doc_images
from agent_customer_support.applications import APPLICATION_NAMES, to_slugs
from agent_customer_support.config import Settings, get_settings
from agent_customer_support.llm import complete_text
from agent_customer_support.llm.normalize import (
    to_anthropic_content,
    to_openai_content,
)
from agent_customer_support.models import AgentResult, QARecord
from agent_customer_support.observability import tracing

logger = logging.getLogger(__name__)

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
        # is a spurious hedge — trust the content and treat it as a valid answer. Measured
        # on the prose only: image markers are ~40 chars each, so counting them could tip
        # a one-line hedge over the threshold and suppress a real miss.
        if len(doc_images.strip(clean)) > 80:
            return clean, None, None
        return clean, "no_answer", None
    return (text or "").strip(), None, None


def _passages_block(passages: list[str]) -> str:
    return "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages))


def _other_applications(metas: list[dict], selected: list[str] | None) -> list[str]:
    """Display names of the applications in `metas` that fall outside `selected`.

    Used only after a widened retry, to tell the user which module actually holds the
    answer they asked for. Compares slugs (what Qdrant stores) and renders display
    names (what the user picked in the widget) — see applications.py on why the two
    forms must not be mixed.

    A passage with no `application` is a global document (the deliberate exception in
    `_build_filter`); it belongs to no module, so it contributes no name rather than a
    guessed one. Order-preserving and deduped, so the note reads in retrieval order.
    """
    scope = set(to_slugs(selected) or [])
    names: dict[str, None] = {}
    for m in metas:
        slug = m.get("application")
        if not slug or slug in scope:
            continue
        names[APPLICATION_NAMES.get(slug, slug)] = None
    return list(names)


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

        # Labelled so this query-rewrite lands as `llm.knowledge.contextualize`,
        # separate from the compose call in `_compose` (`llm.knowledge`) -- it runs on
        # a different model and must not be judged as if it were an answer.
        with tracing.step("contextualize"):
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
        qa_passages: list[str] | None = None,
        qa_leads: bool = False,
        other_applications: list[str] | None = None,
        selected_applications: list[str] | None = None,
    ) -> str:
        """Compose a grounded answer from the always-on process + retrieved passages.

        When CS-verified Q&A passages are present, switch to the three-source prompt
        and append a CS-answer block — marked authoritative when qa_leads, else
        supplementary. With no qa_passages, behavior is identical to the two-source
        path (default).
        """
        qa_passages = qa_passages or []
        if _HAS_PRIOR_TURN in transcript:
            history = f"Lịch sử hội thoại:\n{transcript}\n\n"
        else:
            history = ""
        content = (
            f"{history}Câu hỏi hiện tại: {question}\n\nĐoạn trích:\n{_passages_block(passages)}"
        )
        if qa_passages:
            header = (
                "ĐÁP ÁN CS XÁC NHẬN — ưu tiên cao nhất cho câu hỏi này:"
                if qa_leads
                else "ĐÁP ÁN CS XÁC NHẬN — bổ trợ:"
            )
            content = f"{content}\n\n{header}\n{_passages_block(qa_passages)}"
            compose_prompt = KNOWLEDGE_COMPOSE_PROMPT_WITH_QA
        else:
            compose_prompt = KNOWLEDGE_COMPOSE_PROMPT
        # Both halves or neither: the note's sentence contrasts what the user picked
        # with where the passages actually came from, so it is meaningless without a
        # selection to name. Guarding here also keeps the join off a None default.
        if other_applications and selected_applications:
            content = f"{content}\n\n" + KNOWLEDGE_OTHER_APPLICATION_NOTE.format(
                selected_applications=", ".join(selected_applications),
                other_applications=", ".join(other_applications),
            )
        if not allow_clarify:
            content = f"{content}\n\n{KNOWLEDGE_RESUME_NO_CLARIFY}"
        return complete_text(
            messages=[{"role": "user", "content": content}],
            system=[PROCESS_BLOCK, {"type": "text", "text": compose_prompt}],
            model=cfg.model_for("knowledge"),
        )

    async def _safe_qa_search(
        self, ctx: TurnContext, query: str, applications: list[str] | None, cfg: Settings
    ) -> dict:
        """Search the curated Q&A collection, degrading to an empty result when the
        store is unavailable. The qa collection does not exist until the first CS
        approval, so a missing collection (or a Qdrant outage) must never break the
        guide path.

        Only store-level failures are caught: ApiException covers Qdrant HTTP and
        connection errors, ValueError is what local/in-memory mode raises for a
        missing collection. A bug in our own call would previously be swallowed here
        and silently degrade every answer, so it is left to propagate."""
        try:
            return await ctx.rag.search(
                query,
                collection=cfg.qa_collection,
                # applications=applications,
                top_k=1,
                score_threshold=cfg.qa_lead_threshold,
                # QA is always global and each record is its own short document, so
                # collapsing per source document would wrongly drop distinct records.
                per_doc=None,
            )
        except (ApiException, ValueError) as exc:
            logger.warning("qa search failed, using product-only: %s", exc)
            return {"passages": [], "citations": [], "top_confidence": 0.0}

    async def _with_images(
        self, ctx: TurnContext, passages: list[str], metas: list[dict]
    ) -> tuple[list[str], dict[str, set[str]]]:
        """Rewrite the guides' `media/…` refs into scoped, whitelisted image markers.

        Returns the rewritten passages and the catalog they were rewritten against. The
        catalog is not a by-product: it is the whitelist the composed reply is checked
        against afterwards, so a model that invents an image number cannot get a URL
        signed for a file that does not exist.

        Only the product passages go through this — Q&A records are CS-authored prose and
        carry no refs. Availability is looked up per application, and only for documents
        whose passages actually contain a ref, so image-less guides cost nothing.

        Never raises: without a store handle (or on any store trouble) the refs are simply
        dropped and the answer is text-only, which is the same outcome as a document whose
        media has not been uploaded yet.
        """
        catalog: dict[str, set[str]] = {}
        if ctx.doc_images:
            slugs = doc_images.slugs_with_refs(passages, metas)
            if slugs:
                catalog = await ctx.doc_images.catalog(slugs)
        return doc_images.rewrite_passages(passages, metas, catalog), catalog

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
        # A wrong module selection is a hard filter, not a ranking penalty, so it
        # returns nothing for a question the corpus can answer. Retry across everything
        # this customer is entitled to — never wider than that, or we would explain a
        # module they cannot see in their UI.
        res = await ctx.rag.search_with_fallback(
            query,
            collection=cfg.product_collection,
            applications=applications,
            fallback_applications=ctx.customer.enabled_applications or None,
        )
        passages = res.get("passages", []) or []
        citations = res.get("citations", []) or []
        passages, image_catalog = await self._with_images(ctx, passages, res.get("metas", []) or [])

        qa_res = await self._safe_qa_search(ctx, query, applications, cfg)
        qa_passages = qa_res.get("passages", []) or []
        qa_leads = (
            bool(qa_passages) and (qa_res.get("top_confidence") or 0.0) >= cfg.qa_lead_threshold
        )
        qa_citations = qa_res.get("citations", []) or []
        if qa_citations:
            citations = citations + [f"qa:{c}" for c in qa_citations]

        # Only after a widened retry is there a mismatch worth naming; on a normal hit
        # this stays empty and the compose prompt is byte-identical to before.
        other_applications = (
            _other_applications(res.get("metas", []) or [], applications)
            if res.get("fallback_used")
            else []
        )

        # Always compose: the process context is always in the system prefix, so even
        # with no retrieved passages the model can answer process-level questions.
        # The [[no_answer]] marker is the single miss signal — emitted only when
        # neither the process nor the passages can answer.
        composed = await self._compose(
            query,
            passages,
            ctx.transcript,
            cfg,
            allow_clarify=not already_clarified,
            qa_passages=qa_passages,
            qa_leads=qa_leads,
            other_applications=other_applications,
            selected_applications=ctx.session.selected_applications,
        )
        clean, kind, application = parse_markers(composed)
        # Enforce the image contract on whatever the composer produced: only markers this
        # turn's passages actually offered survive, deduped and capped. Checked against the
        # same catalog the passages were rewritten against, so an invented image number is
        # dropped rather than signed. Done here rather than in _finish so a hallucinated
        # image never reaches the persisted turn.
        clean = doc_images.select(clean, image_catalog, cfg.max_reply_images)

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

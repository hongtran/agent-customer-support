import logging
import re

from qdrant_client.http.exceptions import ApiException

from agent_customer_support.agents.context import TurnContext
from agent_customer_support.agents.passages import passages_block
from agent_customer_support.agents.prompts import (
    KNOWLEDGE_CONTEXTUALIZE_PROMPT,
    KNOWLEDGE_CONTEXTUALIZE_VISION_PROMPT,
    KNOWLEDGE_COMPOSE_PROMPT,
    KNOWLEDGE_COMPOSE_PROMPT_WITH_QA,
    KNOWLEDGE_OTHER_APPLICATION_NOTE,
    KNOWLEDGE_REPAIR_INSTRUCTION,
    KNOWLEDGE_REPAIR_PROMPT,
    KNOWLEDGE_RESUME_NO_CLARIFY,
    KNOWLEDGE_USER_ERROR_NOTE,
    PROCESS_BLOCK,
)
from agent_customer_support import citations as cite
from agent_customer_support import doc_images
from agent_customer_support.applications import APPLICATION_NAMES, to_slugs
from agent_customer_support.config import Settings, get_settings
from agent_customer_support.llm import complete_structured, complete_text
from agent_customer_support.llm.normalize import (
    to_anthropic_content,
    to_openai_content,
)
from agent_customer_support.llm.schemas import ComposedAnswer
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

    The composer reports its status as a schema field now, so this survives for the
    one caller that has no schema to read: the free-text retry in `_compose`, when
    constrained decoding produced nothing.
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


def _resolve_status(composed: ComposedAnswer) -> tuple[str, str, str]:
    """Return (clean_answer, status, application) from a composed reply.

    Two corrections are applied to what the model declared, in Python:

    The hedge rule. A model that writes a full answer and ALSO reports "no_answer"
    is hedging, not missing — trust the content. Measured on the prose only, because
    image markers are ~40 chars each and counting them could tip a one-line hedge
    over the threshold and suppress a real miss.

    And the markers are scrubbed from `answer` regardless. A model told to report its
    status in a field may still write [[clarify]] into the prose out of habit; where
    the two disagree the field wins, but neither may leak to the user.
    """
    clean = _scrub_markers(composed.answer or "")
    status = composed.status
    if status == "no_answer" and len(doc_images.strip(clean)) > 80:
        status = "answer"
    return clean, status, (composed.application or "").strip()


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
        user_error_hint: str = "",
    ) -> ComposedAnswer:
        """Compose a grounded answer from the always-on process + retrieved passages.

        When CS-verified Q&A passages are present, switch to the three-source prompt
        and append a CS-answer block — marked authoritative when qa_leads, else
        supplementary. With no qa_passages, behavior is identical to the two-source
        path (default).

        Returns the reply, the status it decided, and the sources it says it used.
        Only the [[img:...]] markers still ride inline in the prose, because they are
        positional; the control markers are the `status` field.
        """
        qa_passages = qa_passages or []
        if _HAS_PRIOR_TURN in transcript:
            history = f"Lịch sử hội thoại:\n{transcript}\n\n"
        else:
            history = ""
        content = (
            f"{history}Câu hỏi hiện tại: {question}\n\n"
            f"Đoạn trích:\n{passages_block(passages, with_sections=True)}"
        )
        if qa_passages:
            header = (
                "ĐÁP ÁN CS XÁC NHẬN — ưu tiên cao nhất cho câu hỏi này:"
                if qa_leads
                else "ĐÁP ÁN CS XÁC NHẬN — bổ trợ:"
            )
            content = f"{content}\n\n{header}\n{passages_block(qa_passages)}"
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
        # Verification has already ruled this a user error, so the answer must explain
        # the correct usage — and must not send the turn back for a second opinion.
        if user_error_hint:
            content = f"{content}\n\n" + KNOWLEDGE_USER_ERROR_NOTE.format(hint=user_error_hint)
        messages = [{"role": "user", "content": content}]
        system: list[dict] = [PROCESS_BLOCK, {"type": "text", "text": compose_prompt}]
        model = cfg.model_for("knowledge")

        composed = complete_structured(
            messages=messages,
            system=system,
            model=model,
            schema=ComposedAnswer,
        )
        if composed is not None:
            return composed

        # Constrained decoding produced nothing usable — a refusal, a truncation, or an
        # API error. Retry as plain text and ship the answer uncited rather than losing
        # it: contextualize and retrieval have already been paid for, and an answer
        # without its source list is strictly better than no answer. Same trade-off
        # Coordinator._store_attachments makes when S3 is down.
        logger.warning("compose returned no structured answer, retrying as free text")
        # No schema on this path, so the status has to come back out of the prose the
        # old way — this is the one caller `parse_markers` still exists for.
        clean, kind, application = parse_markers(
            complete_text(messages=messages, system=system, model=model) or ""
        )
        return ComposedAnswer(
            answer=clean,
            status=kind or "answer",  # type: ignore[arg-type]
            application=application or "",
            cited=[],
        )

    async def repair(
        self, reply: str, claims: list[dict], source_passages: list[str]
    ) -> str | None:
        """Rewrite `reply` so the flagged minor claims match this turn's sources.

        The second rung of the coordinator's repair ladder: the guardrail found only
        MINOR unsupported claims, and `guardrail.apply_claims` refused to delete them in
        Python (a whole sentence, or a span it could not find exactly once). One call,
        no retry loop here -- the coordinator judges the result once more and escalates
        if it still fails.

        The model may delete or reword, never add. That last rule is enforced in code
        for the one thing that would be expensive to get wrong: an image marker the
        original did not carry is dropped, because presigning an invented one renders
        a broken picture (the same guard `doc_images.select` applies to compose).
        Citations are not touched -- the caller keeps the original answer's, since a
        repair cannot have drawn on a source the answer did not.

        Returns None when the model produced nothing, so the caller can escalate.
        """
        # The judge's own replacement rides along as a hint: apply_claims refused it
        # (too much removed, or a span not found exactly once), it was not judged wrong.
        lines = []
        for c in claims:
            line = f'- "{c.get("span", "")}": {c.get("reason", "")}'
            if c.get("replacement"):
                line += f' (gợi ý sửa: "{c["replacement"]}")'
            lines.append(line)
        claims_block = "\n".join(lines)
        content = (
            f"NGUỒN:\n{passages_block(source_passages)}"
            f"\n\nCÂU TRẢ LỜI:\n{reply}"
            f"\n\nÝ THIẾU CĂN CỨ:\n{claims_block}"
            f"\n\n{KNOWLEDGE_REPAIR_INSTRUCTION}"
        )
        # `llm.knowledge.repair`: distinguishable from the compose call in a trace, the
        # same way contextualize is.
        with tracing.step("repair"):
            raw = complete_text(
                messages=[{"role": "user", "content": content}],
                system=[PROCESS_BLOCK, {"type": "text", "text": KNOWLEDGE_REPAIR_PROMPT}],
                model=get_settings().model_for("knowledge"),
            )
        repaired = (raw or "").strip()
        if not repaired:
            return None
        # The original reply's markers are the whole catalog a repair may draw on.
        allowed: dict[str, set[str]] = {}
        for _kind, slug, name in doc_images.markers_in(reply):
            allowed.setdefault(slug, set()).add(name)
        return doc_images.select(repaired, allowed, get_settings().max_reply_images).strip()

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
            return {"passages": [], "citations": [], "metas": [], "top_confidence": 0.0}

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

    async def run(self, ctx: TurnContext, *, allow_clarify: bool = True) -> AgentResult:
        """
        Single-attempt pipeline:
          contextualize → search → compose (process always-on) → return result

        `allow_clarify` is the caller's: the coordinator owns `clarify_count` and the
        session flag, so this agent decides only WHAT to say, never how many times it
        is allowed to say it. With clarifying disallowed, a miss stops asking and
        becomes a logged `no_answer` instead.

        On a miss (compose reports status="no_answer" — neither process nor passages
        answer) the first attempt asks ONE clarifying question and invites a
        screenshot, so the user can pin down a vague or jargon-y request; once
        clarifying is used up, the miss is logged to the backlog and handed off.
        suspected_bug returns immediately without backlog.
        """
        cfg = get_settings()
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
        passages, image_catalog = await self._with_images(ctx, passages, res.get("metas", []) or [])

        qa_res = await self._safe_qa_search(ctx, query, applications, cfg)
        qa_passages = qa_res.get("passages", []) or []
        qa_leads = (
            bool(qa_passages) and (qa_res.get("top_confidence") or 0.0) >= cfg.qa_lead_threshold
        )

        # Every source the composer is allowed to name this turn. Built BEFORE compose
        # because it is the whitelist the composer's answer is checked against
        # afterwards — the same relationship image_catalog has to the image markers.
        cite_catalog = cite.catalog(res.get("metas", []) or [], qa_res.get("metas", []) or [])

        # Only after a widened retry is there a mismatch worth naming; on a normal hit
        # this stays empty and the compose prompt is byte-identical to before.
        other_applications = (
            _other_applications(res.get("metas", []) or [], applications)
            if res.get("fallback_used")
            else []
        )

        # Always compose: the process context is always in the system prefix, so even
        # with no retrieved passages the model can answer process-level questions.
        # status="no_answer" is the single miss signal — reported only when neither the
        # process nor the passages can answer.
        composed = await self._compose(
            query,
            passages,
            ctx.transcript,
            cfg,
            allow_clarify=allow_clarify,
            qa_passages=qa_passages,
            qa_leads=qa_leads,
            other_applications=other_applications,
            selected_applications=ctx.session.selected_applications,
            user_error_hint=ctx.route_hint,
        )
        clean, status, application = _resolve_status(composed)
        # Enforce the image contract on whatever the composer produced: only markers this
        # turn's passages actually offered survive, deduped and capped. Checked against the
        # same catalog the passages were rewritten against, so an invented image number is
        # dropped rather than signed. Done here rather than in _finish so a hallucinated
        # image never reaches the persisted turn.
        clean = doc_images.select(clean, image_catalog, cfg.max_reply_images)
        # Same enforcement for sources: only ids this turn actually offered survive. A
        # declared id we cannot resolve is dropped rather than shown, because a citation
        # the user cannot check is worse than no citation at all.
        citations = cite.select(composed.cited, cite_catalog, passages, qa_passages)
        # The grounding judge sees EVERY passage this turn retrieved, not only the cited
        # ones: a composer that cites the wrong chunk, or forgets one it used, would
        # otherwise get correct claims flagged. The gate is unchanged — the judge runs
        # only when the answer cited at least one real passage (not the process block,
        # not an invented id); every other answer still skips the call.
        cited_any = cite.passages_for(composed.cited, cite_catalog, passages, qa_passages)
        source_passages = [*passages, *qa_passages] if cited_any else []

        if status == "suspected_bug":
            return AgentResult(
                reply=clean,
                knowledge_status="suspected_bug",
                # `query` is the contextualized one: the verifier's doc check searches
                # the guides again, and this retrieves better than the raw message.
                evidence={
                    "application": application or None,
                    "summary": ctx.message,
                    "query": query,
                },
                citations=citations,
                source_passages=source_passages,
            )

        # Clarify / confirm before answering. The composer judged that an element it
        # can't see (ambiguous subject, unknown user-state, unverified premise, or a
        # risky intent) materially changes the answer. Ask, and let the coordinator
        # count it. With clarifying already used up the compose prompt says not to reach
        # here at all; if the model disobeys, downgrade to a plain answer rather than
        # asking a question the router would only turn into a handoff.
        if status == "clarify":
            if allow_clarify:
                return AgentResult(reply=clean, knowledge_status="clarify", citations=citations)
            return AgentResult(
                reply=clean,
                knowledge_status="answer",
                citations=citations,
                source_passages=source_passages,
            )

        if status != "no_answer":
            return AgentResult(
                reply=clean,
                knowledge_status="answer",
                citations=citations,
                source_passages=source_passages,
            )

        # Miss. While we may still clarify, try to disambiguate before giving up to a
        # human: the request may just be vague or use the customer's own terminology.
        if allow_clarify:
            return AgentResult(
                reply="Mình chưa rõ ý bạn lắm. Bạn cho mình biết cụ thể hơn đang thao tác "
                "ở màn hình/chức năng nào nhé — hoặc chụp giúp mình ảnh màn hình "
                "đang xem để mình hỗ trợ nhanh hơn.",
                knowledge_status="clarify",
            )

        # Out of clarifying attempts and still nothing: genuinely not in the KB — log
        # and hand off.
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
            knowledge_status="no_answer",
        )

import base64
import logging

from agent_customer_support import image_urls
from agent_customer_support.agents.context import TurnContext
from agent_customer_support.applications import to_slug
from agent_customer_support.agents.escalation import EscalationAgent
from agent_customer_support.agents.guardrail import GuardrailAgent, apply_claims
from agent_customer_support.agents.knowledge import KnowledgeAgent
from agent_customer_support.agents.prompts import (
    ASK_CONTACT_REPLY,
    CONTACT_THANKS_REPLY,
    OUT_OF_SCOPE_REPLY,
)
from agent_customer_support.agents.triage import TriageAgent
from agent_customer_support.agents import routing
from agent_customer_support.agents.issue_verification import (
    IssueVerificationAgent,
    fallback_report,
)
from agent_customer_support.config import get_settings
from agent_customer_support.contact import describe, parse as parse_contact
from agent_customer_support.escalation import Escalator
from agent_customer_support.llm.schemas import BugReport
from agent_customer_support.mantis import MantisClient, MantisFile
from agent_customer_support.models import (
    AgentResult,
    Attachment,
    AttachmentRef,
    ChatResponse,
    ContactInfo,
    CustomerProfile,
    StoredAttachment,
    Turn,
    VerifyContext,
)
from agent_customer_support.observability import tracing
from agent_customer_support.rag_client import RagClient
from agent_customer_support.stores.attachment_store import AttachmentStore
from agent_customer_support.stores.conversation_store import ConversationStore
from agent_customer_support.stores.customer_registry import CustomerRegistry
from agent_customer_support.stores.doc_image_store import DocImageStore
from agent_customer_support.stores.flow_store import FlowStore
from agent_customer_support.stores.request_backlog import RequestBacklog
from agent_customer_support.stores.qa_store import QAStore
from agent_customer_support.stores.session_store import SessionStore

logger = logging.getLogger(__name__)

_BLOCK_REPLY = (
    "Xin lỗi, mình chưa thể xử lý nội dung này. Bạn vui lòng nhập câu hỏi về phần mềm CenLab nhé."
)
_FALLBACK_REPLY = "Xin lỗi, mình cần kiểm tra lại thông tin này. Bạn vui lòng hỏi lại sau hoặc yêu cầu gặp nhân viên hỗ trợ."


# Earlier screenshots re-shown to the verifier each turn.
_MAX_PRIOR_IMAGES = 2

_EVIDENCE_EXT = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}


def _evidence_name(index: int, media_type: str) -> str:
    return f"screenshot-{index}.{_EVIDENCE_EXT.get(media_type, 'bin')}"


def _with_slots(report: BugReport, verify: VerifyContext) -> BugReport:
    """Fold the collected slots, and the ones still missing, into the ticket body.

    Naming what is missing is worth as much to the engineer as naming what was
    collected: a ticket filed at the collection cap then says so on its face, rather
    than reading like a complete report that happens to be thin.
    """
    collected = verify.slots.describe()
    if verify.has_image:
        line = "- ảnh chụp màn hình: có (đính kèm)"
        collected = f"{collected}\n{line}" if collected else line
    blocks = []
    if collected:
        blocks.append(f"Thông tin đã thu thập:\n{collected}")
    # What the guides say should happen, next to what the user saw: the engineer can
    # tell at a glance whether the ticket is a real deviation.
    if verify.doc_expected.strip():
        blocks.append(f"Theo tài liệu: {verify.doc_expected.strip()}")
    missing = verify.slots.missing()
    if missing:
        blocks.append("Thiếu thông tin: " + ", ".join(missing))
    if not blocks:
        return report
    return report.model_copy(update={"summary": report.summary + "\n\n" + "\n\n".join(blocks)})


class Coordinator:
    def __init__(self) -> None:
        self.customers = CustomerRegistry()
        self.conversations = ConversationStore()
        self.flow_store = FlowStore()
        self.backlog = RequestBacklog()
        self.qa_store = QAStore()
        self.attachments = AttachmentStore()
        self.doc_images = DocImageStore()
        self.sessions = SessionStore()
        self.rag = RagClient()
        self.escalator = Escalator()
        self.mantis = MantisClient()
        self.guardrail = GuardrailAgent()
        self.triage = TriageAgent()
        self.knowledge = KnowledgeAgent()
        self.issue_verification = IssueVerificationAgent()
        self.escalation = EscalationAgent()

    async def _traced(
        self, name: str, run_coro, ctx: TurnContext, reason: str | None = None
    ) -> AgentResult:
        """Run a sub-agent inside a child span (no-op when tracing is off).

        `agent_span` also labels every LLM generation made inside with this agent's
        name, which is what makes a Langfuse evaluator targetable at one agent.

        `reason` is why the router sent the turn here. It rides as span metadata so a
        trace answers "why did this conversation go to verification?" without anyone
        having to replay it.
        """
        with tracing.agent_span(
            name,
            input={"message": ctx.message},
            metadata={"handoff_reason": reason} if reason else None,
        ) as sp:
            res = await run_coro()
            sp.update(output=res.model_dump(mode="json"))
            return res

    async def handle_turn(
        self,
        *,
        customer_id: str,
        conversation_id: str,
        message: str,
        attachments: list,
        applications: list[str] | None = None,
    ) -> ChatResponse:
        # Root of the turn. session_id groups a whole conversation across turns.
        with tracing.trace(
            "turn", session_id=conversation_id, user_id=customer_id, input=message
        ) as turn:
            # 1. Load context
            customer = await self.customers.get(customer_id) or CustomerProfile(
                customer_id=customer_id, name=customer_id
            )
            session = await self.sessions.get(conversation_id)

            # Store application selection provided at conversation start (first turn only,
            # but allow UI to update it any time a non-empty list is sent).
            if applications:
                session.selected_applications = applications
            conv = await self.conversations.load(conversation_id)
            transcript = "\n".join(f"{t.role}: {t.content}" for t in conv.turns)
            ctx = TurnContext(
                customer=customer,
                session=session,
                conversation=conv,
                message=message,
                attachments=attachments,
                transcript=transcript + f"\nuser: {message}",
                rag=self.rag,
                doc_images=self.doc_images,
                flow_store=self.flow_store,
                backlog=self.backlog,
                qa_store=self.qa_store,
                escalator=self.escalator,
            )

            # 2. Input guardrail
            gin = await self.guardrail.check_input(message)
            if not gin["pass"]:
                resp = await self._finish(ctx, AgentResult(reply=_BLOCK_REPLY), session)
                turn.update(output={"reply": resp.reply, "blocked": True})
                return resp

            # 3-6. Route
            result = await self._route(ctx, session)

            # 7. Output guardrail — grounding only, and only for a reply that cited
            # something; it is then judged against every passage the turn retrieved.
            # `source_passages` is empty for every other route (flow,
            # escalation, out_of_scope) and for knowledge replies that cited nothing, so
            # check_output short-circuits there without an LLM call.
            gout = await self.guardrail.check_output(result.reply, result.source_passages)
            repaired: str | None = None
            if not gout["pass"]:
                result, repaired = await self._repair_or_escalate(ctx, result, gout)
            if result.escalated:
                self._arm_contact_gate(result, session)
            resp = await self._finish(ctx, result, session)
            turn.update(
                output={
                    "reply": resp.reply,
                    "escalated": resp.escalated,
                    "repaired": repaired,
                    "path": ctx.route_path,
                }
            )
            return resp

    async def _repair_or_escalate(
        self, ctx: TurnContext, result: AgentResult, gout: dict
    ) -> tuple[AgentResult, str | None]:
        """Rescue a flagged reply when the judge found only minor problems; else hand off.

        Three rungs, cheapest first, and the second tag says which one rescued it:
          1. "python"  — every claim is minor and `apply_claims` can delete each span
             safely. No further judge call: Python removed exactly the text the judge
             named, so re-judging would spend a call to confirm the judge's own list.
          2. "llm"     — every claim is minor but a span was not safely deletable (a
             whole sentence, or text not found exactly once). One repair call in
             KnowledgeAgent, then the judge sees the repaired reply once more.
          3. None      — any critical claim, no named claim at all, or a repair that
             still fails: escalate exactly as before.

        Citations survive a repair: a repair may delete or reword, never add, so the
        sources still vouch for what remains. They are dropped only on escalation,
        where they would vouch for text the user never sees.
        """
        claims = gout.get("unsupported_claims") or []
        # if only_minor(claims):
        stripped = apply_claims(result.reply, claims)
        if stripped is not None:
            logger.info("ungrounded reply repaired in python: %s", gout.get("reason"))
            result.reply = stripped
            return result, "python"
        with tracing.agent_span("knowledge", input={"claims": claims}) as sp:
            fixed = await self.knowledge.repair(result.reply, claims, result.source_passages)
            sp.update(output={"repaired": fixed})
        if fixed:
            recheck = await self.guardrail.check_output(fixed, result.source_passages)
            if recheck["pass"]:
                logger.info("ungrounded reply repaired by llm: %s", gout.get("reason"))
                result.reply = fixed
                return result, "llm"
        logger.warning("ungrounded reply, escalating: %s", gout.get("reason"))
        # Hand off rather than dead-end. We already know the composed answer cannot be
        # trusted, and the citations belonged to that answer — dropping them keeps a
        # source list from vouching for text the user never sees.
        result = await self._traced(
            "escalation",
            lambda: self.escalation.run(ctx, reason="ungrounded answer"),
            ctx,
        )
        result.reply = _FALLBACK_REPLY
        result.citations = []
        return result, None

    async def _route(self, ctx: TurnContext, session) -> AgentResult:
        """Run steps until one of them ends the turn.

        The decision is `routing.next_step`, a pure function over a small snapshot.
        Everything here is the effects half: running an agent, moving the session
        counters, and folding what came back into the snapshot the next decision
        reads. Keeping the two apart is what makes every routing rule testable
        without a mock, and it is why no agent writes `session.pending` any more —
        the driver owns the flags it later has to reason about.
        """
        contact, refs = self._take_contact(ctx, session)
        self._maybe_cancel(ctx, session)

        state = routing.RouteState.start(session, contact_found=contact is not None)
        result = AgentResult()
        while True:
            step, reason = routing.next_step(state)
            ctx.route_path.append(step)
            if step in routing.TERMINAL:
                return await self._terminal_step(step, ctx, session, result, reason, contact, refs)
            result = await self._agent_step(step, ctx, session, result, reason)
            result.handoff_reason = reason
            state = state.after(result, session)

    def _take_contact(self, ctx: TurnContext, session) -> tuple[ContactInfo | None, dict]:
        """Consume the contact gate, if armed, and read what the user left.

        The flag is consumed either way -- asked once, never again -- and a message
        with no contact in it is simply routed like any other, because it is usually a
        new question and swallowing it would lose it.
        """
        if session.pending != "collect_contact":
            return None, {}
        refs = dict(session.pending_context or {})
        session.pending = None
        session.pending_context = None
        parsed = parse_contact(ctx.message)
        return (parsed if parsed.found else None), refs

    def _maybe_cancel(self, ctx: TurnContext, session) -> None:
        """Let the user walk away from the flow they are in.

        A regex, never a second triage call: this runs on every turn that has a flow
        open, and paying for an LLM call to notice "thôi bỏ qua" would be the most
        expensive way to read two words. Abandoning a flow resets its counters too --
        the next message is a new topic and must not inherit the old one's budget.
        """
        if session.pending not in ("verify_issue", "knowledge_clarify"):
            return
        if not routing.wants_cancel(ctx.message):
            return
        logger.info("user cancelled pending %s", session.pending)
        session.pending = None
        session.pending_context = None
        session.verify_turns = 0
        session.clarify_count = 0

    async def _agent_step(
        self, step: str, ctx: TurnContext, session, prior: AgentResult, reason: str
    ) -> AgentResult:
        """Run one non-terminal step. The loop continues after these."""
        if step == "triage":
            return await self._traced("triage", lambda: self.triage.run(ctx), ctx, reason)
        if step == "knowledge":
            return await self._step_knowledge(ctx, session, prior, reason)
        if step == "issue_verification":
            return await self._step_issue_verification(ctx, session, prior, reason)
        raise AssertionError(f"unroutable step: {step}")

    async def _terminal_step(
        self,
        step: str,
        ctx: TurnContext,
        session,
        prior: AgentResult,
        reason: str,
        contact: ContactInfo | None,
        refs: dict,
    ) -> AgentResult:
        """Run the step that ends the turn and produce the reply the user sees."""
        if step == "reply":
            prior.handoff_reason = reason
            return prior
        if step == "out_of_scope":
            # Refused before any RAG or compose spend. Triage is the single scope gate
            # by design -- KnowledgeAgent stays scope-free to keep its status logic
            # simple, so anything triage lets through gets a normal answer attempt.
            return AgentResult(reply=OUT_OF_SCOPE_REPLY, out_of_scope=True, handoff_reason=reason)
        if step == "attach_contact":
            assert contact is not None  # routing only picks this step when one was found
            res = await self._attach_contact(ctx, session, refs, contact)
            res.handoff_reason = reason
            return res
        if step == "file_ticket":
            return await self._step_file_ticket(ctx, session, reason)
        # escalate. The router's reason IS the handoff reason CS reads, so every rule
        # that gives up on a turn explains itself without a second vocabulary.
        res = await self._traced(
            "escalation", lambda: self.escalation.run(ctx, reason=reason), ctx, reason
        )
        res.handoff_reason = reason
        return res

    async def _step_knowledge(
        self, ctx: TurnContext, session, prior: AgentResult, reason: str
    ) -> AgentResult:
        if session.pending == "knowledge_clarify":
            # This turn is the answer to the question we asked, so the flag is spent.
            session.pending = None
        if prior.verify_outcome == "user_error":
            # Verification's own words on what the user did wrong, so the answer can
            # explain the correct usage instead of starting from nothing.
            ctx.route_hint = prior.reply
        allow_clarify = session.clarify_count < routing.MAX_CLARIFY
        kn = await self._traced(
            "knowledge",
            lambda: self.knowledge.run(ctx, allow_clarify=allow_clarify),
            ctx,
            reason,
        )
        if kn.knowledge_status == "clarify":
            session.clarify_count += 1
            session.pending = "knowledge_clarify"
        return kn

    async def _step_issue_verification(
        self, ctx: TurnContext, session, prior: AgentResult, reason: str
    ) -> AgentResult:
        if session.pending != "verify_issue":
            session.pending = "verify_issue"
            session.pending_context = self._new_verify_context(ctx, prior).model_dump()
            session.verify_turns = 0
        # Earlier screenshots, so the verifier can still read them. From the start of
        # the conversation, not only since the bug was suspected: the user often sends
        # the error screenshot with the question that Knowledge answered first. Capped
        # because each one is paid for in image tokens.
        ctx.evidence_images = (await self._stored_images(ctx, 0))[-_MAX_PRIOR_IMAGES:]
        res = await self._traced(
            "issue_verification", lambda: self.issue_verification.run(ctx), ctx, reason
        )
        session.verify_turns += 1
        # The agent hands back the whole merged context, slots included, so the next
        # turn resumes from what this one learned. Written on every outcome, not only a
        # complete one -- that omission is why no slot state could survive a turn.
        if res.evidence:
            session.pending_context = res.evidence
        if res.verify_outcome == "user_error":
            # Not a bug: close the flow, and remember the verdict, so knowledge
            # suspecting one again reads as a disagreement rather than new evidence.
            session.user_error_seen = True
            session.pending = None
            session.pending_context = None
            session.verify_turns = 0
        return res

    def _new_verify_context(self, ctx: TurnContext, prior: AgentResult) -> VerifyContext:
        """Arm the evidence flow, identically from either way in.

        KnowledgeAgent's suspected_bug result already carries the two keys; a direct
        triage route has none, so they come from the turn itself. Both land here, so
        the ticket path downstream cannot tell which route filled it. `application` is
        a slug, and only when exactly one module is selected, since a ticket cannot
        name two -- MantisBT renders None as "(không rõ)".
        """
        ev = prior.evidence or {}
        selected = ctx.session.selected_applications or []
        application = ev.get("application") or (
            to_slug(selected[0]) if len(selected) == 1 else None
        )
        return VerifyContext(
            application=application,
            summary=str(ev.get("summary") or ctx.message),
            query=str(ev.get("query") or ""),
            # Index the current user turn will get once _finish persists it. The ticket
            # attaches screenshots from this turn onward only, so an unrelated image
            # sent earlier in the same conversation never lands on the bug.
            since_turn=len(ctx.conversation.turns),
        )

    async def _step_file_ticket(self, ctx: TurnContext, session, reason: str) -> AgentResult:
        """Ticket, then backlog row, then handoff.

        MantisBT goes first so the backlog row can carry the ticket id in its single
        put_item; it never raises, so a tracker outage costs the ticket and nothing
        after it.

        Reached two ways and deliberately identical in both: the verifier said
        `bug_confirmed`, or the collection hit its turn cap. The cap case has no
        BugReport of its own, so one is derived from the opening message -- filing a
        thinner ticket instead of trapping the user is the entire point of the cap.
        """
        verify = VerifyContext.model_validate(session.pending_context or {})
        report = (
            BugReport.model_validate(verify.report)
            if verify.report
            else fallback_report(verify.summary or ctx.message)
        )
        report = _with_slots(report, verify)
        files = await self._evidence_files(ctx, verify.since_turn)
        issue = await self.mantis.create_issue(
            report=report,
            customer_id=ctx.customer.customer_id,
            customer_name=ctx.customer.name,
            application=verify.application,
            transcript=ctx.transcript,
            files=files,
        )
        rec = await self.backlog.add(
            customer_id=ctx.customer.customer_id,
            type="bug",
            summary=report.summary,
            title=report.title,
            application=verify.application,
            transcript=ctx.transcript,
            mantis_issue_id=issue.id if issue else None,
            mantis_issue_url=issue.url if issue else None,
        )
        session.pending = None
        session.pending_context = None
        session.verify_turns = 0
        note = (
            f"Ticket MantisBT: {issue.url}"
            if issue
            else "Ticket MantisBT: KHÔNG tạo được (lỗi kết nối), vui lòng tạo tay."
        )
        esc = await self._traced(
            "escalation",
            lambda: self.escalation.run(ctx, reason="verified bug", note=note),
            ctx,
            reason,
        )
        esc.handoff_reason = reason
        # So a contact left on the next turn can be attached to what was just filed.
        esc.escalation_refs = {
            "backlog_id": rec.id,
            "mantis_issue_id": issue.id if issue else None,
            "mantis_issue_url": issue.url if issue else None,
        }
        return esc

    def _arm_contact_gate(self, result: AgentResult, session) -> None:
        """Append the contact ask to a handoff reply and remember what it belongs to.

        Runs once per handoff, in handle_turn, so every path that sets `escalated`
        (triage, knowledge, issue verification and the guardrail fallback) gets the same
        behaviour without knowing about it. There is exactly one session object per
        turn -- the driver mutates it in place and `_finish` saves it -- so this writes
        straight to it.
        """
        session.pending = "collect_contact"
        session.pending_context = {
            "reason": result.escalation_reason,
            **(result.escalation_refs or {}),
        }
        result.reply = (
            f"{result.reply}\n\n{ASK_CONTACT_REPLY}" if result.reply else ASK_CONTACT_REPLY
        )

    async def _attach_contact(
        self, ctx: TurnContext, session, refs: dict, contact: ContactInfo
    ) -> AgentResult:
        """Write the contact onto everything the handoff created, then tell CS again.

        Every step is best-effort and independent: the handoff already went out, so a
        store or tracker problem here must cost that one update, never the reply or
        the other updates.
        """
        customer_id = ctx.customer.customer_id
        try:
            await self.conversations.set_contact(ctx.session.conversation_id, customer_id, contact)
        except Exception as exc:  # noqa: BLE001 - degrade
            logger.warning("conversation contact update failed: %s", exc)
        if refs.get("backlog_id"):
            try:
                await self.backlog.set_contact(refs["backlog_id"], contact)
            except Exception as exc:  # noqa: BLE001 - degrade
                logger.warning("backlog contact update failed: %s", exc)
        if refs.get("mantis_issue_id"):
            # add_note never raises
            await self.mantis.add_note(
                refs["mantis_issue_id"],
                f"Liên hệ khách hàng: {describe(contact)}\n{contact.raw}",
            )
        try:
            await self.escalator.contact_update(
                customer_id=customer_id,
                customer_name=ctx.customer.name,
                reason=refs.get("reason"),
                contact=contact,
                ticket_url=refs.get("mantis_issue_url"),
            )
        except Exception as exc:  # noqa: BLE001 - degrade
            logger.warning("CS contact notification failed: %s", exc)
        return AgentResult(reply=CONTACT_THANKS_REPLY)

    async def _evidence_files(self, ctx: TurnContext, since_turn: int | None) -> list[MantisFile]:
        """Screenshots for the ticket: earlier issue-verification turns, then the current one.

        Earlier turns hold only S3 keys, so those are read back; the current turn still
        carries its base64 and costs nothing. Never raises — a screenshot that cannot
        be read is dropped and the ticket is filed without it. Capped to the most
        recent `mantis_max_files`, because a long back-and-forth can hold many images
        and the newest are the ones that made the evidence complete.
        """
        files = [
            MantisFile(name=_evidence_name(i, att.media_type), content_b64=att.data)
            for i, att in enumerate(
                [*await self._stored_images(ctx, since_turn), *ctx.attachments], start=1
            )
        ]
        cap = get_settings().mantis_max_files
        return files[-cap:] if cap > 0 else []

    async def _stored_images(self, ctx: TurnContext, since_turn: int | None) -> list[Attachment]:
        """Screenshots the user sent on earlier turns since `since_turn`, read back from S3.

        Shared by the ticket (`_evidence_files`) and the verifier, which is re-shown
        them each turn so a screen name or an error dialog can still be read after the
        turn it arrived on. User turns only. Never raises — a screenshot that cannot be
        read is dropped, and the ticket or the turn goes on without it.
        """
        start = since_turn if since_turn is not None else len(ctx.conversation.turns)
        images: list[Attachment] = []
        for turn in ctx.conversation.turns[start:]:
            if turn.role != "user":
                continue
            for stored in turn.attachments:
                try:
                    raw = await self.attachments.get_bytes(stored)
                except Exception as exc:  # noqa: BLE001 - one lost screenshot, not the turn
                    logger.warning("evidence read failed for %s: %s", stored.s3_key, exc)
                    continue
                images.append(
                    Attachment(
                        kind="image",
                        media_type=stored.media_type,
                        data=base64.b64encode(raw).decode(),
                    )
                )
        return images

    async def _finish(self, ctx: TurnContext, result: AgentResult, session) -> ChatResponse:
        session.conversation_id = ctx.session.conversation_id
        await self.sessions.save(session)

        # Persist turns. The user turn is built first so its id can key the S3 objects.
        user_turn = Turn(role="user", content=ctx.message)
        user_turn.attachments = await self._store_attachments(ctx, user_turn.id)
        await self.conversations.append(
            ctx.session.conversation_id,
            ctx.customer.customer_id,
            user_turn,
        )
        assistant_turn = Turn(role="assistant", content=result.reply)
        await self.conversations.append(
            ctx.session.conversation_id,
            ctx.customer.customer_id,
            assistant_turn,
        )
        return ChatResponse(
            conversation_id=ctx.session.conversation_id,
            reply=await self._resolve_images(result.reply),
            escalated=result.escalated,
            citations=result.citations,
            message_id=assistant_turn.id,
            attachments=await self._presign(user_turn.attachments),
        )

    async def _resolve_images(self, reply: str) -> str:
        """Swap image markers for presigned URLs, for the response only.

        The turn was already persisted above with markers intact — deliberately, because a
        presigned URL expires: storing one would archive a dead link and feed 500
        characters of signature into the transcript the LLM re-reads next turn. Same
        split as StoredAttachment/AttachmentRef, and it means re-rendering history later
        is just re-signing.

        Runs after the output guardrail so the guardrail judges prose, not signatures.
        Never raises: a signing problem drops the picture, never the answer.
        """
        return await image_urls.resolve_doc_images(self.doc_images, reply)

    async def _store_attachments(self, ctx: TurnContext, turn_id: str) -> list[StoredAttachment]:
        """Upload this turn's images to S3 and return their keys.

        Never raises. By the time _finish runs, the reply has already been generated —
        contextualize, retrieval and compose have all been paid for — so an S3 problem
        must not turn a good answer into a 500. Losing a screenshot from the archive is
        the strictly cheaper failure. (This is the same mistake that made the original
        DynamoDB size error user-visible.)
        """
        if not ctx.attachments:
            return []
        try:
            return [
                await self.attachments.put(ctx.session.conversation_id, turn_id, i, a)
                for i, a in enumerate(ctx.attachments)
            ]
        except Exception as exc:  # noqa: BLE001 - degrade, never break a generated reply
            logger.warning("attachment upload failed, persisting turn without it: %s", exc)
            return []

    async def _presign(self, stored: list[StoredAttachment]) -> list[AttachmentRef]:
        """Signed URLs so the widget can render what was just uploaded. Also
        best-effort: a signing failure costs a thumbnail, not the answer."""
        return await image_urls.presign_attachments(self.attachments, stored)

import logging

from agent_customer_support import doc_images
from agent_customer_support.agents.context import TurnContext
from agent_customer_support.agents.escalation import EscalationAgent
from agent_customer_support.agents.flow import FlowAgent
from agent_customer_support.agents.guardrail import GuardrailAgent
from agent_customer_support.agents.knowledge import KnowledgeAgent
from agent_customer_support.agents.prompts import OUT_OF_SCOPE_REPLY
from agent_customer_support.agents.triage import TriageAgent
from agent_customer_support.agents.verification import IssueVerificationAgent
from agent_customer_support.escalation import Escalator
from agent_customer_support.models import (
    AgentResult,
    AttachmentRef,
    ChatResponse,
    CustomerProfile,
    StoredAttachment,
    Turn,
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
_FALLBACK_REPLY = "Xin lỗi, mình cần kiểm tra lại thông tin này. Bạn vui lòng thử lại hoặc yêu cầu gặp nhân viên hỗ trợ."


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
        self.guardrail = GuardrailAgent()
        self.triage = TriageAgent()
        self.knowledge = KnowledgeAgent()
        self.flow = FlowAgent()
        self.verification = IssueVerificationAgent()
        self.escalation = EscalationAgent()

    async def _traced(self, name: str, run_coro, ctx: TurnContext) -> AgentResult:
        """Run a sub-agent inside a child span (no-op when tracing is off).

        `agent_span` also labels every LLM generation made inside with this agent's
        name, which is what makes a Langfuse evaluator targetable at one agent.
        """
        with tracing.agent_span(name, input={"message": ctx.message}) as sp:
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

            # 7. Output guardrail
            # gout = await self.guardrail.check_output(result.reply)
            # if not gout["pass"]:
            #     result = AgentResult(
            #         reply=_FALLBACK_REPLY,
            #         escalated=result.escalated,
            #         new_session=result.new_session,
            #     )
            resp = await self._finish(ctx, result, session)
            turn.update(output={"reply": resp.reply, "escalated": resp.escalated})
            return resp

    async def _route(self, ctx: TurnContext, session) -> AgentResult:
        # Resume pending verification
        if session.pending == "verify_issue":
            res = await self._traced("verification", lambda: self.verification.run(ctx), ctx)
            return await self._after_verification(ctx, res, session)

        # Resume pending knowledge clarification: the user is answering our clarify
        # question (often with a screenshot), so go straight back to knowledge —
        # bypass triage to keep the loop deterministic and bounded to one attempt.
        if session.pending == "knowledge_clarify":
            return await self._knowledge_phase(ctx, session)

        # Active flow
        if session.active_flow_id:
            return await self._traced("flow", lambda: self.flow.run(ctx), ctx)

        # Triage (route-only)
        tri = await self._traced("triage", lambda: self.triage.run(ctx), ctx)
        if tri.routed_to == "flow":
            return await self._traced("flow", lambda: self.flow.run(ctx), ctx)
        if tri.routed_to == "escalate":
            return await self._traced(
                "escalation", lambda: self.escalation.run(ctx, reason="user requested human"), ctx
            )
        # Clearly off-topic: refuse before any RAG or compose spend. Triage is the
        # single scope gate by design — KnowledgeAgent stays scope-free to keep its
        # marker logic simple, so anything triage lets through gets a normal answer
        # attempt (and at worst the no_answer/clarify path).
        if tri.routed_to == "out_of_scope":
            return AgentResult(reply=OUT_OF_SCOPE_REPLY, resolved=True, out_of_scope=True)

        return await self._knowledge_phase(ctx, session)

    async def _knowledge_phase(self, ctx: TurnContext, session) -> AgentResult:
        kn = await self._traced("knowledge", lambda: self.knowledge.run(ctx), ctx)
        if kn.suspected_bug:
            session.pending = "verify_issue"
            session.pending_context = kn.evidence
            ver = await self._traced("verification", lambda: self.verification.run(ctx), ctx)
            return await self._after_verification(ctx, ver, session)
        if kn.resolved is False:
            return await self._traced(
                "escalation", lambda: self.escalation.run(ctx, reason="knowledge unresolved"), ctx
            )
        # resolved, or a clarify reply (resolved is None) — return as-is.
        return kn

    async def _after_verification(self, ctx, res: AgentResult, session) -> AgentResult:
        if not res.evidence_complete:
            # Keep pending; session already has pending="verify_issue" set.
            # Return verification result as-is (new_session=None so _finish uses session).
            return res
        # Evidence ready -> log bug + escalate
        ev = res.evidence or {}
        await self.backlog.add(
            customer_id=ctx.customer.customer_id,
            type="bug",
            summary=ev.get("summary", "bug"),
            application=ev.get("application"),
            transcript=ctx.transcript,
        )
        session.pending = None
        session.pending_context = None
        esc = await self._traced(
            "escalation", lambda: self.escalation.run(ctx, reason="verified bug"), ctx
        )
        esc.new_session = session
        return esc

    async def _finish(self, ctx: TurnContext, result: AgentResult, session) -> ChatResponse:
        # Apply session changes: prefer result.new_session if provided, else use session
        new_session = result.new_session if result.new_session is not None else session
        new_session.conversation_id = ctx.session.conversation_id
        await self.sessions.save(new_session)

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
        try:
            urls = {}
            for kind, slug, name in doc_images.markers_in(reply):
                urls[(slug, name)] = await self.doc_images.presign(slug, name)
            if not urls:
                return reply
            return doc_images.presign_markers(reply, lambda s, n: urls[(s, n)])
        except Exception as exc:  # noqa: BLE001 - degrade, never break a generated reply
            logger.warning("doc image presign failed, replying without images: %s", exc)
            return doc_images.strip(reply)

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
        if not stored:
            return []
        try:
            return [await self.attachments.presign(s) for s in stored]
        except Exception as exc:  # noqa: BLE001 - degrade, never break a generated reply
            logger.warning("presign failed, returning reply without image urls: %s", exc)
            return []

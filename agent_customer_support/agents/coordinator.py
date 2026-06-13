from agent_customer_support.agents.context import TurnContext
from agent_customer_support.agents.escalation import EscalationAgent
from agent_customer_support.agents.flow import FlowAgent
from agent_customer_support.agents.guardrail import GuardrailAgent
from agent_customer_support.agents.knowledge import KnowledgeAgent
from agent_customer_support.agents.triage import TriageAgent
from agent_customer_support.agents.verification import IssueVerificationAgent
from agent_customer_support.escalation import Escalator
from agent_customer_support.models import (
    AgentResult, ChatResponse, CustomerProfile, Turn,
)
from agent_customer_support.observability import tracing
from agent_customer_support.rag_client import RagClient
from agent_customer_support.stores.conversation_store import ConversationStore
from agent_customer_support.stores.customer_registry import CustomerRegistry
from agent_customer_support.stores.flow_store import FlowStore
from agent_customer_support.stores.request_backlog import RequestBacklog
from agent_customer_support.stores.session_store import SessionStore

_BLOCK_REPLY = "Xin lỗi, mình chưa thể xử lý nội dung này. Bạn vui lòng nhập câu hỏi về phần mềm CenLab nhé."
_FALLBACK_REPLY = "Xin lỗi, mình cần kiểm tra lại thông tin này. Bạn vui lòng thử lại hoặc yêu cầu gặp nhân viên hỗ trợ."


class Coordinator:
    def __init__(self) -> None:
        self.customers = CustomerRegistry()
        self.conversations = ConversationStore()
        self.flow_store = FlowStore()
        self.backlog = RequestBacklog()
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
        """Run a sub-agent inside a child span (no-op when tracing is off)."""
        with tracing.span(f"agent.{name}", input={"message": ctx.message}) as sp:
            res = await run_coro()
            sp.update(output=res.model_dump(mode="json"))
            return res

    async def handle_turn(self, *, customer_id: str, conversation_id: str,
                          message: str, attachments: list) -> ChatResponse:
        # Root of the turn. session_id groups a whole conversation across turns.
        with tracing.trace("turn", session_id=conversation_id, user_id=customer_id,
                           input=message) as turn:
            # 1. Load context
            customer = await self.customers.get(customer_id) or CustomerProfile(
                customer_id=customer_id, name=customer_id)
            session = await self.sessions.get(conversation_id)
            conv = await self.conversations.load(conversation_id)
            transcript = "\n".join(f"{t.role}: {t.content}" for t in conv.turns)
            ctx = TurnContext(
                customer=customer, session=session, conversation=conv,
                message=message, attachments=attachments,
                transcript=transcript + f"\nuser: {message}",
                rag=self.rag, flow_store=self.flow_store,
                backlog=self.backlog, escalator=self.escalator,
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
            gout = await self.guardrail.check_output(result.reply)
            if not gout["pass"]:
                result = AgentResult(reply=_FALLBACK_REPLY,
                                     escalated=result.escalated,
                                     new_session=result.new_session)
            resp = await self._finish(ctx, result, session)
            turn.update(output={"reply": resp.reply, "escalated": resp.escalated})
            return resp

    async def _route(self, ctx: TurnContext, session) -> AgentResult:
        # Resume pending verification
        if session.pending == "verify_issue":
            res = await self._traced("verification", lambda: self.verification.run(ctx), ctx)
            return await self._after_verification(ctx, res, session)

        # Active flow
        if session.active_flow_id:
            return await self._traced("flow", lambda: self.flow.run(ctx), ctx)

        # Triage
        tri = await self._traced("triage", lambda: self.triage.run(ctx), ctx)
        if tri.action == "reply":
            return tri
        if tri.routed_to == "flow":
            return await self._traced("flow", lambda: self.flow.run(ctx), ctx)
        if tri.routed_to == "escalate":
            return await self._traced(
                "escalation",
                lambda: self.escalation.run(ctx, reason="user requested human"), ctx)

        # knowledge
        kn = await self._traced("knowledge", lambda: self.knowledge.run(ctx), ctx)
        if kn.suspected_bug:
            session.pending = "verify_issue"
            session.pending_context = kn.evidence
            ver = await self._traced("verification", lambda: self.verification.run(ctx), ctx)
            return await self._after_verification(ctx, ver, session)
        if kn.resolved is False:
            return await self._traced(
                "escalation",
                lambda: self.escalation.run(ctx, reason="knowledge unresolved"), ctx)
        return kn

    async def _after_verification(self, ctx, res: AgentResult, session) -> AgentResult:
        if not res.evidence_complete:
            # Keep pending; session already has pending="verify_issue" set.
            # Return verification result as-is (new_session=None so _finish uses session).
            return res
        # Evidence ready -> log bug + escalate
        ev = res.evidence or {}
        await self.backlog.add(
            customer_id=ctx.customer.customer_id, type="bug",
            summary=ev.get("summary", "bug"), module=ev.get("module"),
            transcript=ctx.transcript)
        session.pending = None
        session.pending_context = None
        esc = await self._traced(
            "escalation", lambda: self.escalation.run(ctx, reason="verified bug"), ctx)
        esc.new_session = session
        return esc

    async def _finish(self, ctx: TurnContext, result: AgentResult,
                      session) -> ChatResponse:
        # Apply session changes: prefer result.new_session if provided, else use session
        new_session = result.new_session if result.new_session is not None else session
        new_session.conversation_id = ctx.session.conversation_id
        await self.sessions.save(new_session)

        # Persist turns
        await self.conversations.append(
            ctx.session.conversation_id, ctx.customer.customer_id,
            Turn(role="user", content=ctx.message, attachments=ctx.attachments))
        await self.conversations.append(
            ctx.session.conversation_id, ctx.customer.customer_id,
            Turn(role="assistant", content=result.reply))
        return ChatResponse(conversation_id=ctx.session.conversation_id,
                            reply=result.reply, escalated=result.escalated,
                            citations=result.citations)

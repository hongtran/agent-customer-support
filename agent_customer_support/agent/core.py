import json
import re
from agent_customer_support.llm import complete_with_tools
from agent_customer_support.models import ChatResponse, CustomerProfile, Turn
from agent_customer_support.rag_client import RagClient
from agent_customer_support.escalation import Escalator
from agent_customer_support.flows.engine import FlowEngine
from agent_customer_support.stores.customer_registry import CustomerRegistry
from agent_customer_support.stores.conversation_store import ConversationStore
from agent_customer_support.stores.flow_store import FlowStore
from agent_customer_support.stores.request_backlog import RequestBacklog
from agent_customer_support.stores.session_store import SessionStore
from agent_customer_support.agent.prompt import build_system_prompt
from agent_customer_support.agent.tools import TOOL_DEFS, ToolContext, dispatch

_GOTO_RE = re.compile(r"\[\[goto:([a-zA-Z0-9_\-]+)\]\]")
MAX_TOOL_ROUNDS = 6


def parse_goto(text: str) -> tuple[str, str | None]:
    m = _GOTO_RE.search(text or "")
    if not m:
        return (text or "", None)
    clean = _GOTO_RE.sub("", text).strip()
    return (clean, m.group(1))


class AgentCore:
    def __init__(self) -> None:
        self.customers = CustomerRegistry()
        self.conversations = ConversationStore()
        self.flow_store = FlowStore()
        self.backlog = RequestBacklog()
        self.sessions = SessionStore()
        self.rag = RagClient()
        self.escalator = Escalator()

    async def handle_turn(
        self, *, customer_id: str, conversation_id: str, user_msg: str
    ) -> ChatResponse:
        customer = await self.customers.get(customer_id) or CustomerProfile(
            customer_id=customer_id, name=customer_id
        )
        session = await self.sessions.get(conversation_id)
        conv = await self.conversations.load(conversation_id)

        active_flow = None
        if session.active_flow_id:
            active_flow = await self.flow_store.get(session.active_flow_id)

        system = build_system_prompt(customer, session, active_flow)
        transcript = "\n".join(f"{t.role}: {t.content}" for t in conv.turns)
        ctx = ToolContext(
            customer=customer,
            rag=self.rag,
            flow_store=self.flow_store,
            backlog=self.backlog,
            escalator=self.escalator,
            conversation_id=conversation_id,
            transcript=transcript + f"\nuser: {user_msg}",
        )

        messages: list[dict] = [{"role": "user", "content": user_msg}]
        escalated = False
        final_text = ""
        for _ in range(MAX_TOOL_ROUNDS):
            out = complete_with_tools(messages=messages, tools=TOOL_DEFS, system=system)
            if out["stop_reason"] != "tool_use":
                final_text = out.get("text") or ""
                break
            messages.append({"role": "assistant", "content": out.get("text") or ""})
            tool_results = []
            for call in out["tool_calls"]:
                result = await dispatch(call["name"], call["input"], ctx)
                if call["name"] == "escalate_to_human":
                    escalated = True
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        clean_text, goto = parse_goto(final_text)
        if active_flow and goto:
            res = FlowEngine.resolve(active_flow, goto)
            if res.kind == "outcome":
                if res.outcome is not None and res.outcome.type == "escalate":
                    await self.escalator.escalate(
                        customer_id=customer_id,
                        reason=res.outcome.reason or "flow escalate",
                        transcript=ctx.transcript,
                    )
                    escalated = True
                session.active_flow_id = None
                session.current_step_id = None
            else:
                if res.step is not None:
                    session.current_step_id = res.step.id
            await self.sessions.save(session)
        final_text = clean_text

        await self.conversations.append(
            conversation_id, customer_id, Turn(role="user", content=user_msg)
        )
        await self.conversations.append(
            conversation_id, customer_id, Turn(role="assistant", content=final_text)
        )
        return ChatResponse(conversation_id=conversation_id, reply=final_text, escalated=escalated)

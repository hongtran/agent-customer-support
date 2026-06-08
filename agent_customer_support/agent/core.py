import json
import logging
import re
import textwrap
from agent_customer_support.config import get_settings
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

logger = logging.getLogger(__name__)
_GOTO_RE = re.compile(r"\[\[goto:([a-zA-Z0-9_\-]+)\]\]")
MAX_TOOL_ROUNDS = 6


def _dbg(label: str, data: str = "", width: int = 72) -> None:
    """Print a compact debug block — only active when logger is DEBUG."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    border = "─" * width
    print(f"\n\033[36m┌{border}┐\033[0m")
    print(f"\033[36m│ {label:<{width-1}}\033[0m\033[36m│\033[0m")
    if data:
        print(f"\033[36m├{border}┤\033[0m")
        for line in textwrap.wrap(data, width - 2) or [data]:
            print(f"\033[36m│ {line:<{width-1}}│\033[0m")
    print(f"\033[36m└{border}┘\033[0m")


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

        # ── DEBUG: context ──────────────────────────────────────────────────
        _dbg(
            f"TURN  customer={customer_id}  conv={conversation_id}",
            f"user_msg: {user_msg!r}  |  "
            f"modules={customer.enabled_modules}  |  "
            f"active_flow={session.active_flow_id}  step={session.current_step_id}",
            # f"system_message={system}"
        )

        messages: list[dict] = [{"role": "user", "content": user_msg}]
        escalated = False
        final_text = ""

        for round_n in range(MAX_TOOL_ROUNDS):
            # ── DEBUG: LLM call ─────────────────────────────────────────────
            _dbg(f"LLM CALL  round={round_n + 1}/{MAX_TOOL_ROUNDS}  model={get_settings().agent_model} messages={messages}")

            out = complete_with_tools(messages=messages, tools=TOOL_DEFS, system=system)

            # ── DEBUG: LLM response ─────────────────────────────────────────
            tool_names = [c["name"] for c in out.get("tool_calls", [])]
            _dbg(
                f"LLM RESPONSE  stop_reason={out['stop_reason']}",
                (
                    f"tools_called={tool_names}"
                    if tool_names
                    else f"text={repr((out.get('text') or '')[:120])}"
                ),
            )

            if out["stop_reason"] != "tool_use":
                final_text = out.get("text") or ""
                break

            # Build provider-appropriate assistant message
            model = get_settings().agent_model
            is_anthropic = "claude" in model
            if is_anthropic:
                messages.append({"role": "assistant", "content": out.get("text") or ""})
            else:
                # OpenAI: assistant message must include tool_calls list
                messages.append({
                    "role": "assistant",
                    "content": out.get("text"),
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {
                                "name": c["name"],
                                "arguments": json.dumps(c["input"], ensure_ascii=False),
                            },
                        }
                        for c in out["tool_calls"]
                    ],
                })

            tool_results_anthropic = []
            for call in out["tool_calls"]:
                # ── DEBUG: tool dispatch ────────────────────────────────────
                _dbg(
                    f"TOOL CALL  [{call['name']}]",
                    f"args={json.dumps(call['input'], ensure_ascii=False)}",
                )

                result = await dispatch(call["name"], call["input"], ctx)
                if call["name"] == "escalate_to_human":
                    escalated = True

                # ── DEBUG: tool result ──────────────────────────────────────
                result_preview = json.dumps(result, ensure_ascii=False)
                _dbg(
                    f"TOOL RESULT  [{call['name']}]",
                    result_preview[:300] + ("…" if len(result_preview) > 300 else ""),
                )

                if is_anthropic:
                    tool_results_anthropic.append({
                        "type": "tool_result",
                        "tool_use_id": call["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                else:
                    # OpenAI: each tool result is a separate "tool" role message
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    })

            if is_anthropic:
                messages.append({"role": "user", "content": tool_results_anthropic})

        # ── DEBUG: flow marker ──────────────────────────────────────────────
        clean_text, goto = parse_goto(final_text)

        # Resolve which flow to use: session-active OR just fetched via get_flow tool
        effective_flow = active_flow or ctx.last_fetched_flow

        if goto:
            _dbg(
                f"FLOW MARKER  [[goto:{goto}]]",
                f"active_flow={active_flow and active_flow.id}  "
                f"effective={effective_flow and effective_flow.id}",
            )

        if effective_flow and goto:
            res = FlowEngine.resolve(effective_flow, goto)
            if res.kind == "outcome":
                if res.outcome is not None and res.outcome.type == "escalate":
                    await self.escalator.escalate(
                        customer_id=customer_id,
                        reason=res.outcome.reason or "flow escalate",
                        transcript=ctx.transcript,
                    )
                    escalated = True
                _dbg(f"FLOW OUTCOME  type={res.outcome and res.outcome.type}  → session cleared")
                session.active_flow_id = None
                session.current_step_id = None
            else:
                if res.step is not None:
                    # Activate flow in session (covers both: advancing existing flow
                    # AND starting a new flow that agent just fetched via get_flow)
                    if not session.active_flow_id:
                        session.active_flow_id = effective_flow.id
                        _dbg(f"FLOW ACTIVATED  flow={effective_flow.id}  first_step={res.step.id}")
                    _dbg(f"FLOW ADVANCE  → step={res.step.id}")
                    session.current_step_id = res.step.id
            await self.sessions.save(session)
        final_text = clean_text

        # ── DEBUG: final reply ──────────────────────────────────────────────
        _dbg(
            f"FINAL REPLY  escalated={escalated}",
            repr(final_text[:200]) + ("…" if len(final_text) > 200 else ""),
        )

        await self.conversations.append(
            conversation_id, customer_id, Turn(role="user", content=user_msg)
        )
        await self.conversations.append(
            conversation_id, customer_id, Turn(role="assistant", content=final_text)
        )
        return ChatResponse(conversation_id=conversation_id, reply=final_text, escalated=escalated)

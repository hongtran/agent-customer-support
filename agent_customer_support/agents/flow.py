import json
import re

from agent_customer_support.agents.context import TurnContext
from agent_customer_support.config import get_settings
from agent_customer_support.flows.engine import FlowEngine
from agent_customer_support.llm import complete_with_tools
from agent_customer_support.models import AgentResult
from agent_customer_support.agent.prompt import build_system_prompt
from agent_customer_support.agent.tools import ToolContext, dispatch

_GOTO_RE = re.compile(r"\[\[goto:([a-zA-Z0-9_\-]+)\]\]")


def build_assistant_message(out: dict, is_anthropic: bool) -> dict:
    """Build the assistant turn echoing the model's tool calls for the next round.

    Anthropic: content is a list of text + tool_use blocks (required so the following
    tool_result blocks have matching tool_use). OpenAI: content is the text (may be None)
    plus a tool_calls array.
    """
    if is_anthropic:
        blocks: list[dict] = []
        text = out.get("text")
        if text:
            blocks.append({"type": "text", "text": text})
        for c in out["tool_calls"]:
            blocks.append(
                {"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["input"]}
            )
        return {"role": "assistant", "content": blocks}
    return {
        "role": "assistant",
        "content": out.get("text"),
        "tool_calls": [
            {
                "id": c["id"],
                "type": "function",
                "function": {"name": c["name"], "arguments": json.dumps(c["input"])},
            }
            for c in out["tool_calls"]
        ],
    }


FLOW_TOOLS = [
    {
        "name": "list_flows",
        "description": "Liệt kê các quy trình khả dụng cho khách hàng hiện tại.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_flow",
        "description": "Lấy chi tiết một quy trình theo flow_id để dẫn từng bước.",
        "input_schema": {
            "type": "object",
            "properties": {"flow_id": {"type": "string"}},
            "required": ["flow_id"],
        },
    },
]
MAX_ROUNDS = 5


def parse_goto(text: str) -> tuple[str, str | None]:
    m = _GOTO_RE.search(text or "")
    if not m:
        return (text or "", None)
    return (_GOTO_RE.sub("", text).strip(), m.group(1))


class FlowAgent:
    name = "flow"

    async def run(self, ctx: TurnContext) -> AgentResult:
        session = ctx.session.model_copy(deep=True)
        active_flow = None
        if session.active_flow_id:
            active_flow = await ctx.flow_store.get(session.active_flow_id)

        system = build_system_prompt(ctx.customer, session, active_flow)
        tool_ctx = ToolContext(
            customer=ctx.customer,
            rag=ctx.rag,
            flow_store=ctx.flow_store,
            backlog=ctx.backlog,
            escalator=ctx.escalator,
            conversation_id=session.conversation_id,
            transcript=ctx.transcript,
        )
        messages: list[dict] = [{"role": "user", "content": ctx.message}]
        final_text = ""
        flow_model = get_settings().model_for("flow")
        is_anthropic = "claude" in flow_model

        for _ in range(MAX_ROUNDS):
            out = complete_with_tools(
                messages=messages, tools=FLOW_TOOLS, system=system, model=flow_model
            )
            if out["stop_reason"] != "tool_use":
                final_text = out.get("text") or ""
                break
            messages.append(build_assistant_message(out, is_anthropic))
            if is_anthropic:
                results = []
                for call in out["tool_calls"]:
                    r = await dispatch(call["name"], call["input"], tool_ctx)
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": call["id"],
                            "content": json.dumps(r, ensure_ascii=False),
                        }
                    )
                messages.append({"role": "user", "content": results})
            else:
                for call in out["tool_calls"]:
                    r = await dispatch(call["name"], call["input"], tool_ctx)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": json.dumps(r, ensure_ascii=False),
                        }
                    )

        clean, goto = parse_goto(final_text)
        effective = active_flow or tool_ctx.last_fetched_flow
        escalated = False
        if effective and goto:
            res = FlowEngine.resolve(effective, goto)
            if res.kind == "outcome":
                if res.outcome and res.outcome.type == "escalate":
                    await ctx.escalator.escalate(
                        customer_id=ctx.customer.customer_id,
                        reason=res.outcome.reason or "flow escalate",
                        transcript=ctx.transcript,
                    )
                    escalated = True
                session.active_flow_id = None
                session.current_step_id = None
            elif res.step is not None:
                if not session.active_flow_id:
                    session.active_flow_id = effective.id
                session.current_step_id = res.step.id

        return AgentResult(reply=clean, new_session=session, escalated=escalated)

import re

from agent_customer_support.agents.context import TurnContext
from agent_customer_support.agents.prompts import VERIFICATION_PROMPT
from agent_customer_support.config import get_settings
from agent_customer_support.llm import complete_with_tools
from agent_customer_support.llm.normalize import (
    to_anthropic_content,
    to_openai_content,
)
from agent_customer_support.models import AgentResult

_READY_RE = re.compile(r"\[\[evidence_ready\]\]")


class IssueVerificationAgent:
    name = "verification"

    async def run(self, ctx: TurnContext) -> AgentResult:
        cfg = get_settings()
        model = cfg.model_for("verification")
        is_anthropic = "claude" in model
        # Prior turns as plain text so the agent knows what bug is being verified.
        history = [{"role": t.role, "content": t.content} for t in ctx.conversation.turns]
        # Current turn carries multimodal content (the user may attach a screenshot).
        if is_anthropic:
            current_content = to_anthropic_content(ctx.message, ctx.attachments)
        else:
            current_content = to_openai_content(ctx.message, ctx.attachments)

        out = complete_with_tools(
            messages=history + [{"role": "user", "content": current_content}],
            tools=[],
            system=VERIFICATION_PROMPT,
            model=model,
        )
        text = out.get("text") or ""
        ready = bool(_READY_RE.search(text))
        clean = _READY_RE.sub("", text).strip()

        if ready:
            evidence = dict(ctx.session.pending_context or {})
            evidence["has_image"] = any(a.kind == "image" for a in ctx.attachments)
            return AgentResult(reply=clean, evidence_complete=True, evidence=evidence)
        return AgentResult(reply=clean, evidence_complete=False)

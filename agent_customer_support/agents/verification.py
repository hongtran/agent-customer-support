import re

from agent_customer_support.agents.context import TurnContext
from agent_customer_support.agents.prompts import VERIFICATION_PROMPT
from agent_customer_support.config import get_settings
from agent_customer_support.llm import complete_with_tools
from agent_customer_support.llm.normalize import (
    to_anthropic_content, to_openai_content,
)
from agent_customer_support.models import AgentResult

_READY_RE = re.compile(r"\[\[evidence_ready\]\]")


class IssueVerificationAgent:
    name = "verification"

    async def run(self, ctx: TurnContext) -> AgentResult:
        is_anthropic = "claude" in get_settings().agent_model
        if is_anthropic:
            content = to_anthropic_content(ctx.message, ctx.attachments)
        else:
            content = to_openai_content(ctx.message, ctx.attachments)

        out = complete_with_tools(
            messages=[{"role": "user", "content": content}],
            tools=[], system=VERIFICATION_PROMPT,
        )
        text = out.get("text") or ""
        ready = bool(_READY_RE.search(text))
        clean = _READY_RE.sub("", text).strip()

        if ready:
            evidence = dict(ctx.session.pending_context or {})
            evidence["has_image"] = any(a.kind == "image" for a in ctx.attachments)
            return AgentResult(reply=clean, evidence_complete=True, evidence=evidence)
        return AgentResult(reply=clean, evidence_complete=False)

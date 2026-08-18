import re

from agent_customer_support.agents.context import TurnContext
from agent_customer_support.agents.prompts import TRIAGE_PROMPT
from agent_customer_support.config import get_settings
from agent_customer_support.llm import complete_structured
from agent_customer_support.llm.schemas import TriageDecision
from agent_customer_support.models import AgentResult

_HUMAN_RE = re.compile(r"(gặp|cho).{0,12}(nhân viên|người thật|tư vấn viên|cs)", re.I)


class TriageAgent:
    name = "triage"

    async def run(self, ctx: TurnContext) -> AgentResult:
        # Rule fast-paths (no LLM). Note this is the only way `flow` is ever chosen —
        # which is why TriageDecision does not offer it as a target.
        if ctx.session.active_flow_id:
            return AgentResult(action="route", routed_to="flow")
        if _HUMAN_RE.search(ctx.message or ""):
            return AgentResult(action="route", routed_to="escalate")

        # Triage is route-only: clarification is owned by KnowledgeAgent, which has
        # the RAG context (and screenshots) needed to ask a useful follow-up.
        decision = complete_structured(
            messages=ctx.as_messages(),
            system=TRIAGE_PROMPT,
            model=get_settings().model_for("triage"),
            schema=TriageDecision,
        )
        # Fail-safe default. Constrained decoding removes the malformed-JSON and
        # unknown-target cases, so this now fires only when there is no decision at
        # all — an API error, a refusal, or a truncated response.
        target = decision.target if decision else "knowledge"
        return AgentResult(action="route", routed_to=target)

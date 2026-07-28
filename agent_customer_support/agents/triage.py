import json
import re

from agent_customer_support.agents.context import TurnContext
from agent_customer_support.agents.prompts import TRIAGE_PROMPT
from agent_customer_support.config import get_settings
from agent_customer_support.llm import complete_text
from agent_customer_support.models import AgentResult

_HUMAN_RE = re.compile(r"(gặp|cho).{0,12}(nhân viên|người thật|tư vấn viên|cs)", re.I)


class TriageAgent:
    name = "triage"

    async def run(self, ctx: TurnContext) -> AgentResult:
        # Rule fast-paths (no LLM)
        if ctx.session.active_flow_id:
            return AgentResult(action="route", routed_to="flow")
        if _HUMAN_RE.search(ctx.message or ""):
            return AgentResult(action="route", routed_to="escalate")

        raw = complete_text(
            messages=ctx.as_messages(),
            system=TRIAGE_PROMPT,
            model=get_settings().model_for("triage"),
        )
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            data = {"target": "knowledge"}  # fail-safe default

        # Triage is route-only: clarification is owned by KnowledgeAgent, which has
        # the RAG context (and screenshots) needed to ask a useful follow-up.
        target = data.get("target", "knowledge")
        if target not in ("knowledge", "flow", "escalate"):
            target = "knowledge"
        return AgentResult(action="route", routed_to=target)

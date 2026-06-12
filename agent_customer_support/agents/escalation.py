from agent_customer_support.agents.context import TurnContext
from agent_customer_support.models import AgentResult

_REPLY = "Mình đã chuyển yêu cầu của bạn cho nhân viên hỗ trợ. Bạn vui lòng chờ trong giây lát nhé."


class EscalationAgent:
    name = "escalation"

    async def run(self, ctx: TurnContext, *, reason: str = "escalation") -> AgentResult:
        await ctx.escalator.escalate(
            customer_id=ctx.customer.customer_id,
            reason=reason,
            transcript=ctx.transcript,
        )
        return AgentResult(reply=_REPLY, escalated=True, routed_to="escalate")

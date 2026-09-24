from agent_customer_support.agents.context import TurnContext
from agent_customer_support.models import AgentResult

_REPLY = "Mình đã chuyển yêu cầu của bạn cho nhân viên hỗ trợ. Bạn vui lòng chờ trong giây lát nhé."

# The only handoff reasons that notify CS (Zalo + email, the customer's manager in CC).
# Every other reason ("user requested human", "hop limit", "bug loop", "ungrounded
# answer") only returns the reply: nobody is paged.
NOTIFY_REASONS = frozenset({"verified bug", "knowledge unresolved", "clarify limit"})


class EscalationAgent:
    name = "escalation"

    async def run(
        self, ctx: TurnContext, *, reason: str = "escalation", note: str | None = None
    ) -> AgentResult:
        if reason in NOTIFY_REASONS:
            manager = ctx.customer.manager_email
            await ctx.escalator.escalate(
                customer_id=ctx.customer.customer_id,
                reason=reason,
                transcript=ctx.transcript,
                note=note,
                cc=[manager] if manager else None,
            )
        return AgentResult(
            reply=_REPLY, escalated=True, routed_to="escalate", escalation_reason=reason
        )

from typing import Protocol, runtime_checkable

from agent_customer_support.agents.context import TurnContext
from agent_customer_support.models import AgentResult


@runtime_checkable
class Agent(Protocol):
    name: str

    async def run(self, ctx: TurnContext) -> AgentResult: ...

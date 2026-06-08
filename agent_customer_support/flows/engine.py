from dataclasses import dataclass
from agent_customer_support.models import Flow, FlowStep, FlowOutcome


@dataclass
class Resolved:
    kind: str  # "step" | "outcome"
    step: FlowStep | None = None
    outcome: FlowOutcome | None = None


class FlowEngine:
    @staticmethod
    def first_step(flow: Flow) -> FlowStep:
        if not flow.steps:
            raise ValueError(f"Flow {flow.id} has no steps")
        return flow.steps[0]

    @staticmethod
    def get_step(flow: Flow, step_id: str) -> FlowStep:
        for s in flow.steps:
            if s.id == step_id:
                return s
        raise KeyError(f"Step {step_id} not in flow {flow.id}")

    @staticmethod
    def allowed_gotos(flow: Flow, step_id: str) -> list[str]:
        return [t.goto for t in FlowEngine.get_step(flow, step_id).next]

    @staticmethod
    def resolve(flow: Flow, goto: str) -> Resolved:
        if goto in flow.outcomes:
            return Resolved(kind="outcome", outcome=flow.outcomes[goto])
        for s in flow.steps:
            if s.id == goto:
                return Resolved(kind="step", step=s)
        raise KeyError(f"goto {goto} not found in flow {flow.id}")

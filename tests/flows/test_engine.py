from agent_customer_support.models import Flow, FlowStep, FlowTransition, FlowOutcome
from agent_customer_support.flows.engine import FlowEngine


def _flow():
    return Flow(
        id="f1",
        title="t",
        module="m",
        triggers=["x"],
        steps=[
            FlowStep(
                id="s1",
                say="A",
                next=[FlowTransition(when="ok", goto="s2"), FlowTransition(when="loi", goto="esc")],
            ),
            FlowStep(id="s2", say="B", next=[FlowTransition(when="xong", goto="done")]),
        ],
        outcomes={
            "done": FlowOutcome(type="success", say="bye"),
            "esc": FlowOutcome(type="escalate", reason="khong xu ly duoc"),
        },
    )


def test_first_step():
    assert FlowEngine.first_step(_flow()).id == "s1"


def test_resolve_goto_to_step():
    res = FlowEngine.resolve(_flow(), "s2")
    assert res.kind == "step" and res.step.id == "s2"


def test_resolve_goto_to_outcome():
    res = FlowEngine.resolve(_flow(), "esc")
    assert res.kind == "outcome" and res.outcome.type == "escalate"


def test_allowed_gotos():
    assert set(FlowEngine.allowed_gotos(_flow(), "s1")) == {"s2", "esc"}


def test_get_step():
    assert FlowEngine.get_step(_flow(), "s2").say == "B"

from agent_customer_support.models import (
    CustomerProfile,
    SessionState,
    Flow,
    FlowStep,
    FlowTransition,
)
from agent_customer_support.agent.prompt import build_system_prompt


def test_prompt_includes_modules_and_try_then_route():
    cust = CustomerProfile(customer_id="c1", name="C1", enabled_modules=["xet-nghiem", "quan-trac"])
    p = build_system_prompt(cust, SessionState(conversation_id="cv1"), active_flow=None)
    assert "xet-nghiem" in p and "quan-trac" in p
    assert "log_request" in p
    assert "search_knowledge" in p


def test_prompt_injects_current_flow_step():
    flow = Flow(
        id="f1",
        title="Tạo mẫu",
        module="xet-nghiem",
        triggers=["x"],
        steps=[FlowStep(id="s1", say="Vào menu X", next=[FlowTransition(when="ok", goto="done")])],
    )
    state = SessionState(conversation_id="cv1", active_flow_id="f1", current_step_id="s1")
    p = build_system_prompt(CustomerProfile(customer_id="c1", name="C1"), state, active_flow=flow)
    assert "s1" in p and "Vào menu X" in p
    assert "[[goto:" in p

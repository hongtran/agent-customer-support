from agent_customer_support.models import (
    Flow,
    FlowStep,
    FlowTransition,
    FlowOutcome,
    CustomerProfile,
    SessionState,
    ChatRequest,
)


def test_flow_roundtrip():
    flow = Flow(
        id="f1",
        title="t",
        module="m",
        scope="global",
        version=1,
        language="vi",
        triggers=["x"],
        steps=[FlowStep(id="s1", say="hello", next=[FlowTransition(when="ok", goto="done")])],
        outcomes={"done": FlowOutcome(type="success", say="bye")},
    )
    data = flow.model_dump()
    assert Flow.model_validate(data).steps[0].next[0].goto == "done"


def test_customer_profile_defaults():
    c = CustomerProfile(customer_id="c1", name="Cust 1", enabled_modules=["xet-nghiem"])
    assert c.config_notes is None


def test_session_state():
    s = SessionState(conversation_id="cv1")
    assert s.active_flow_id is None and s.current_step_id is None


def test_chat_request():
    r = ChatRequest(customer_id="c1", conversation_id="cv1", message="hi")
    assert r.message == "hi"

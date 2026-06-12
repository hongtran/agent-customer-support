from agent_customer_support.models import (
    Flow,
    FlowStep,
    FlowTransition,
    FlowOutcome,
    CustomerProfile,
    SessionState,
    ChatRequest,
    Attachment,
    AgentResult,
    Turn,
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


def test_attachment_image():
    a = Attachment(kind="image", media_type="image/png", data="aGVsbG8=")
    assert a.kind == "image"
    assert a.media_type == "image/png"


def test_session_pending_defaults_none():
    s = SessionState(conversation_id="cv1")
    assert s.pending is None
    assert s.pending_context is None


def test_session_pending_verify():
    s = SessionState(conversation_id="cv1", pending="verify_issue",
                     pending_context={"summary": "x", "module": "m"})
    assert s.pending == "verify_issue"
    assert s.pending_context["summary"] == "x"


def test_chat_request_attachments_default_empty():
    r = ChatRequest(customer_id="c1", conversation_id="cv1", message="hi")
    assert r.attachments == []


def test_agent_result_defaults():
    r = AgentResult(reply="hello")
    assert r.action == "reply"
    assert r.routed_to is None
    assert r.suspected_bug is False
    assert r.evidence_complete is False
    assert r.escalated is False


def test_agent_result_route():
    r = AgentResult(action="route", routed_to="knowledge")
    assert r.action == "route"
    assert r.routed_to == "knowledge"


def test_turn_attachments_default_empty():
    t = Turn(role="user", content="hi")
    assert t.attachments == []

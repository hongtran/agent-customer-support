from agent_customer_support.models import (
    BugSlots,
    VerifyContext,
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
        application="m",
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
    assert s.pending is None
    # The routing counters must default, or a session written before they existed
    # would fail to load and lose whatever flow it was in.
    assert (s.clarify_count, s.verify_turns, s.user_error_seen) == (0, 0, False)


def test_session_state_loads_a_row_written_before_the_counters_existed():
    old = '{"conversation_id": "cv1", "pending": "verify_issue"}'
    s = SessionState.model_validate_json(old)
    assert s.pending == "verify_issue" and s.clarify_count == 0


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
    s = SessionState(
        conversation_id="cv1",
        pending="verify_issue",
        pending_context={"summary": "x", "module": "m"},
    )
    assert s.pending == "verify_issue"
    assert s.pending_context["summary"] == "x"


def test_chat_request_attachments_default_empty():
    r = ChatRequest(customer_id="c1", conversation_id="cv1", message="hi")
    assert r.attachments == []


def test_agent_result_defaults():
    r = AgentResult(reply="hello")
    assert r.routed_to is None
    assert r.knowledge_status is None
    assert r.verify_outcome is None
    assert r.handoff_reason is None
    assert r.escalated is False


def test_agent_result_route():
    r = AgentResult(routed_to="knowledge")
    assert r.routed_to == "knowledge"


def test_turn_attachments_default_empty():
    t = Turn(role="user", content="hi")
    assert t.attachments == []


# ---- bug slots ----


def _slots(**kw):
    return BugSlots.empty().model_copy(update=kw)


def test_empty_slots_need_no_arguments_but_the_schema_keeps_every_field_required():
    # Required with "" as the no-value case: a defaulted field is dropped from
    # `required` under OpenAI strict mode, which is why `empty()` exists at all.
    assert BugSlots.empty().steps == ""
    schema = BugSlots.model_json_schema()
    assert set(schema["required"]) == set(BugSlots.model_fields)


def test_merge_keeps_a_filled_slot_when_the_update_says_nothing():
    merged = _slots(steps="1. Bấm Import").merge(_slots(actual="Lỗi 500"))
    assert merged.steps == "1. Bấm Import" and merged.actual == "Lỗi 500"


def test_merge_lets_a_new_value_correct_an_old_one():
    assert _slots(actual="Lỗi 500").merge(_slots(actual="Lỗi 404")).actual == "Lỗi 404"


def test_missing_names_only_the_required_slots():
    assert _slots(steps="s", expected="e", actual="a").missing() == []
    # version and occurred_at are useful, never blocking.
    assert _slots(steps="s", expected="e", actual="a", version="").missing() == []
    assert "kết quả mong đợi" in _slots(steps="s", actual="a").missing()


def test_describe_lists_only_what_was_filled():
    out = _slots(steps="1. Bấm Import").describe()
    assert "1. Bấm Import" in out
    assert "kết quả mong đợi" not in out


def test_the_slot_is_a_module_and_not_the_application():
    """An application is what the customer bought and what scopes Qdrant; a module is
    the screen inside it. One field holding both would scope a search to a screen
    name, which matches nothing."""
    assert "module" in BugSlots.model_fields
    assert "application" not in BugSlots.model_fields
    assert "application" in VerifyContext.model_fields


def test_a_module_slot_does_not_touch_the_context_application():
    v = VerifyContext(application="lay_mau_quan_trac", summary="s")
    v.slots = v.slots.merge(_slots(module="Danh sách phiếu yêu cầu"))
    assert v.application == "lay_mau_quan_trac"
    assert v.slots.module == "Danh sách phiếu yêu cầu"
    assert "màn hình/menu: Danh sách phiếu yêu cầu" in v.slots.describe()


# ---- verify context ----


def test_verify_context_loads_a_row_written_before_slots_existed():
    """A session mid-collection must survive the deploy that added the slots, or the
    evidence already gathered is lost."""
    v = VerifyContext.model_validate(
        {"application": "lay_mau_quan_trac", "summary": "A lỗi", "since_turn": 2}
    )
    assert v.slots == BugSlots.empty()
    assert v.has_image is False and v.report is None


def test_verify_context_ignores_keys_it_does_not_know():
    v = VerifyContext.model_validate({"summary": "s", "something_old": 1})
    assert v.summary == "s"

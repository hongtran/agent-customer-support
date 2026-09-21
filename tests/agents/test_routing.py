"""The routing rules, tested as what they are: a pure function.

No mocks, no fixtures, no event loop — `next_step` takes a frozen snapshot and
returns a step and a reason, so every rule is one assert. That is the entire
reason this logic was lifted out of `Coordinator._route`, where reaching any one
branch meant standing up five agents and nine stores.
"""

import pytest

from agent_customer_support.agents import routing
from agent_customer_support.agents.routing import RouteState, next_step
from agent_customer_support.models import AgentResult, SessionState


def _state(**kw) -> RouteState:
    return RouteState(**kw)


# ---- the opening move ----


def test_a_fresh_turn_starts_at_triage():
    assert next_step(_state()) == ("triage", "new turn")


@pytest.mark.parametrize(
    "target,step",
    [
        ("knowledge", "knowledge"),
        ("issue_verification", "issue_verification"),
        ("escalate", "escalate"),
        ("out_of_scope", "out_of_scope"),
    ],
)
def test_every_triage_target_has_a_step(target, step):
    assert next_step(_state(triage_target=target))[0] == step


# ---- resuming a flow the user is already in ----


def test_pending_verification_resumes_without_triage():
    assert next_step(_state(pending="verify_issue")) == (
        "issue_verification",
        "resume verification",
    )


def test_pending_clarify_goes_straight_back_to_knowledge():
    assert next_step(_state(pending="knowledge_clarify")) == ("knowledge", "clarify answered")


def test_a_pending_flow_outranks_what_triage_would_have_said():
    s = _state(pending="verify_issue", triage_target="knowledge")
    assert next_step(s)[0] == "issue_verification"


def test_a_found_contact_outranks_a_pending_flow():
    s = _state(contact_found=True, pending="verify_issue")
    assert next_step(s) == ("attach_contact", "contact given")


# ---- what knowledge decided ----


def test_an_answer_is_replied():
    assert next_step(_state(knowledge_status="answer")) == ("reply", "answered")


def test_a_clarify_is_replied_and_waits():
    assert next_step(_state(knowledge_status="clarify")) == ("reply", "need clarification")


def test_a_miss_escalates():
    assert next_step(_state(knowledge_status="no_answer")) == (
        "escalate",
        "knowledge unresolved",
    )


def test_a_suspected_bug_opens_verification():
    assert next_step(_state(knowledge_status="suspected_bug")) == (
        "issue_verification",
        "suspected bug",
    )


# ---- what verification decided ----


def test_a_confirmed_bug_files_a_ticket():
    assert next_step(_state(verify_outcome="bug_confirmed")) == ("file_ticket", "verified bug")


def test_a_user_error_goes_back_to_knowledge_and_files_nothing():
    assert next_step(_state(verify_outcome="user_error")) == (
        "knowledge",
        "not a bug, explain usage",
    )


def test_still_collecting_just_replies():
    assert next_step(_state(verify_outcome="need_more_info")) == ("reply", "collecting evidence")


# ---- the limits, which is why this module exists ----


def test_the_hop_limit_escalates_before_anything_else():
    # Even mid-flow with a perfectly good route available.
    s = _state(hop_count=routing.MAX_HOPS, pending="verify_issue", knowledge_status="answer")
    assert next_step(s) == ("escalate", "hop limit")


def test_the_collection_cap_files_what_it_has_instead_of_asking_again():
    s = _state(pending="verify_issue", verify_turns=routing.MAX_VERIFY_TURNS)
    assert next_step(s) == ("file_ticket", "verify turn cap")


def test_one_turn_under_the_cap_still_collects():
    s = _state(pending="verify_issue", verify_turns=routing.MAX_VERIFY_TURNS - 1)
    assert next_step(s)[0] == "issue_verification"


def test_the_clarify_limit_escalates_rather_than_asking_a_third_time():
    s = _state(knowledge_status="clarify", clarify_count=routing.MAX_CLARIFY)
    assert next_step(s) == ("escalate", "clarify limit")


def test_one_clarify_under_the_limit_still_asks():
    s = _state(knowledge_status="clarify", clarify_count=routing.MAX_CLARIFY - 1)
    assert next_step(s)[0] == "reply"


def test_knowledge_re_suspecting_a_bug_after_a_user_error_escalates():
    # Otherwise knowledge and verification hand the turn back and forth until the
    # hop limit, paying for a model call each way.
    s = _state(knowledge_status="suspected_bug", user_error_seen=True)
    assert next_step(s) == ("escalate", "bug loop")


# ---- the snapshot ----


def test_start_reads_the_session_counters():
    session = SessionState(
        conversation_id="c1",
        pending="verify_issue",
        clarify_count=1,
        verify_turns=3,
        user_error_seen=True,
    )
    s = RouteState.start(session, contact_found=True)
    assert (s.pending, s.clarify_count, s.verify_turns, s.user_error_seen, s.contact_found) == (
        "verify_issue",
        1,
        3,
        True,
        True,
    )


def test_after_clears_pending_so_a_step_is_not_re_entered():
    session = SessionState(conversation_id="c1")
    s = RouteState.start(SessionState(conversation_id="c1", pending="verify_issue"))
    s = s.after(AgentResult(verify_outcome="need_more_info"), session)
    assert s.pending is None
    assert next_step(s)[0] == "reply"


def test_after_counts_the_hop_and_keeps_the_triage_route():
    session = SessionState(conversation_id="c1")
    s = RouteState.start(session).after(AgentResult(routed_to="knowledge"), session)
    assert (s.hop_count, s.triage_target) == (1, "knowledge")
    # A later step that routes nowhere must not erase it.
    s = s.after(AgentResult(knowledge_status="answer"), session)
    assert (s.hop_count, s.triage_target) == (2, "knowledge")


def test_terminal_covers_every_step_that_produces_a_reply():
    # A step missing from TERMINAL would be handed to _agent_step and raise.
    assert routing.TERMINAL == {
        "escalate",
        "out_of_scope",
        "attach_contact",
        "file_ticket",
        "reply",
    }


# ---- cancelling a flow ----


@pytest.mark.parametrize(
    "message",
    [
        "thôi bỏ qua đi",
        "Thôi, mình hỏi việc khác",
        "bỏ qua vụ này nhé",
        "quên đi, mình có câu hỏi khác",
        "không cần nữa",
        "khoan đã",
        "hủy",
    ],
)
def test_cancel_phrases_are_recognised(message):
    assert routing.wants_cancel(message)


@pytest.mark.parametrize(
    "message",
    [
        "mình chỉ cần in phiếu thôi",
        "làm xong rồi thì bấm lưu thôi",
        "cho mình hỏi cách tạo phiếu yêu cầu",
        "lỗi này xảy ra khi bấm nút khác",
        "",
    ],
)
def test_ordinary_messages_are_not_cancels(message):
    # A false positive throws away collected evidence, so the bare verbs only count
    # at the start of a message.
    assert not routing.wants_cancel(message)

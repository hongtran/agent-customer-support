from agent_customer_support.agents.prompts import (
    TRIAGE_PROMPT,
    ISSUE_VERIFICATION_PROMPT,
    GROUNDING_JUDGE_PROMPT,
    KNOWLEDGE_CONTEXTUALIZE_PROMPT,
    KNOWLEDGE_COMPOSE_PROMPT,
    PROCESS_CONTEXT,
    PROCESS_BLOCK,
)


def test_prompts_are_nonempty_strings():
    for p in (
        TRIAGE_PROMPT,
        ISSUE_VERIFICATION_PROMPT,
        GROUNDING_JUDGE_PROMPT,
        KNOWLEDGE_CONTEXTUALIZE_PROMPT,
        KNOWLEDGE_COMPOSE_PROMPT,
        PROCESS_CONTEXT,
    ):
        assert isinstance(p, str) and len(p) > 20


def test_triage_is_route_only():
    # Triage no longer clarifies; it only routes to knowledge/escalate.
    assert "knowledge" in TRIAGE_PROMPT.lower()
    assert "escalate" in TRIAGE_PROMPT.lower()
    assert "clarify" not in TRIAGE_PROMPT.lower()


def test_compose_asks_for_every_status_by_name():
    # The four values of ComposedAnswer.status. A status the prompt never names is
    # one the model will never choose, and the router would never see.
    for status in ("answer", "clarify", "no_answer", "suspected_bug"):
        assert f'"{status}"' in KNOWLEDGE_COMPOSE_PROMPT


def test_compose_no_longer_asks_for_control_markers_in_the_prose():
    # These moved to the `status` field. Leaving them in the prompt would have the
    # model write them into `answer`, where they would be scrubbed and the status
    # silently lost.
    for marker in ("[[clarify]]", "[[no_answer]]", "[[suspected_bug:"):
        assert marker not in KNOWLEDGE_COMPOSE_PROMPT


def test_compose_still_asks_for_image_markers_inline():
    # Unlike the control markers, these are positional and stay in the prose.
    assert "[[img:" in KNOWLEDGE_COMPOSE_PROMPT


def test_compose_has_no_out_of_scope_marker():
    # Scope validation lives in triage only, by design — KnowledgeAgent's marker
    # set stays small. If someone re-adds [[out_of_scope]] here, parse_markers
    # would silently leave it in user-facing replies.
    assert "[[out_of_scope]]" not in KNOWLEDGE_COMPOSE_PROMPT


def test_triage_prompt_offers_out_of_scope():
    assert "out_of_scope" in TRIAGE_PROMPT


def test_compose_forbids_exposing_internal_refs():
    # Step codes and passage indices must not leak to the user.
    assert "KHÔNG lộ tham chiếu nội bộ" in KNOWLEDGE_COMPOSE_PROMPT


def test_compose_routes_admin_cases():
    # Permission / missing-master-data questions must be flagged as admin work.
    assert "VIỆC THUỘC ADMIN" in KNOWLEDGE_COMPOSE_PROMPT


def test_process_context_has_admin_section_with_guidance():
    assert "VIỆC THUỘC QUẢN TRỊ HỆ THỐNG/ADMIN" in PROCESS_CONTEXT
    # the two canonical guidance lines for admin-owned cases (permission + master data)
    assert "liên hệ quản trị hệ thống/admin" in PROCESS_CONTEXT
    assert "chuẩn hoá master data" in PROCESS_CONTEXT


def test_process_block_is_cacheable():
    assert PROCESS_BLOCK["type"] == "text"
    assert PROCESS_BLOCK["text"] == PROCESS_CONTEXT
    assert PROCESS_BLOCK["cache_control"] == {"type": "ephemeral"}


def test_verification_prompt_names_every_slot_and_outcome():
    from agent_customer_support.models import BugSlots

    for slot in BugSlots.model_fields:
        assert slot in ISSUE_VERIFICATION_PROMPT, slot
    for outcome in ("need_more_info", "user_error", "bug_confirmed"):
        assert outcome in ISSUE_VERIFICATION_PROMPT, outcome


def test_verification_prompt_caps_how_much_it_asks_for_at_once():
    # The whole point of slot filling is asking for a little at a time; a prompt
    # that dumps the full checklist on the user is how a collection gets abandoned.
    assert "TỐI ĐA HAI" in ISSUE_VERIFICATION_PROMPT


def test_verification_no_longer_uses_the_evidence_ready_marker():
    assert "evidence_ready" not in ISSUE_VERIFICATION_PROMPT

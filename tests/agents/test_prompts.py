from agent_customer_support.agents.prompts import (
    TRIAGE_PROMPT,
    VERIFICATION_PROMPT,
    GUARDRAIL_OUTPUT_PROMPT,
    KNOWLEDGE_CONTEXTUALIZE_PROMPT,
    KNOWLEDGE_COMPOSE_PROMPT,
    PROCESS_CONTEXT,
    PROCESS_BLOCK,
)


def test_prompts_are_nonempty_strings():
    for p in (
        TRIAGE_PROMPT,
        VERIFICATION_PROMPT,
        GUARDRAIL_OUTPUT_PROMPT,
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


def test_compose_has_no_answer_and_bug_markers():
    assert "[[no_answer]]" in KNOWLEDGE_COMPOSE_PROMPT
    assert "suspected_bug" in KNOWLEDGE_COMPOSE_PROMPT


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

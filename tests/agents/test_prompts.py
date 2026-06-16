from agent_customer_support.agents.prompts import (
    TRIAGE_PROMPT,
    VERIFICATION_PROMPT,
    GUARDRAIL_OUTPUT_PROMPT,
    KNOWLEDGE_GRADER_PROMPT,
    KNOWLEDGE_REFORMULATE_PROMPT,
    KNOWLEDGE_COMPOSE_PROMPT,
    DIAGNOSTIC_PROMPT,
)
from agent_customer_support.config import Settings


def test_prompts_are_nonempty_strings():
    for p in (
        TRIAGE_PROMPT,
        VERIFICATION_PROMPT,
        GUARDRAIL_OUTPUT_PROMPT,
        KNOWLEDGE_GRADER_PROMPT,
        KNOWLEDGE_REFORMULATE_PROMPT,
        KNOWLEDGE_COMPOSE_PROMPT,
    ):
        assert isinstance(p, str) and len(p) > 20


def test_triage_is_route_only():
    # Triage no longer clarifies; it only routes to knowledge/escalate.
    assert "knowledge" in TRIAGE_PROMPT.lower()
    assert "escalate" in TRIAGE_PROMPT.lower()
    assert "clarify" not in TRIAGE_PROMPT.lower()


def test_grader_judges_content_not_score():
    assert "answer_present" in KNOWLEDGE_GRADER_PROMPT


def test_compose_has_no_answer_and_bug_markers():
    assert "[[no_answer]]" in KNOWLEDGE_COMPOSE_PROMPT
    assert "suspected_bug" in KNOWLEDGE_COMPOSE_PROMPT


def test_diagnostic_prompt_requests_rule_id_json():
    assert "rule_id" in DIAGNOSTIC_PROMPT
    assert "JSON" in DIAGNOSTIC_PROMPT


def test_diagnostic_model_setting_overridable():
    s = Settings(diagnostic_model="claude-haiku-4-5-20251001")
    assert s.model_for("diagnostic") == "claude-haiku-4-5-20251001"
    # falls back to agent_model when unset
    assert Settings(diagnostic_model=None).model_for("diagnostic") == Settings().agent_model

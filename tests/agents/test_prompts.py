from agent_customer_support.agents.prompts import (
    TRIAGE_PROMPT, VERIFICATION_PROMPT, GUARDRAIL_OUTPUT_PROMPT,
    KNOWLEDGE_GRADER_PROMPT, KNOWLEDGE_REFORMULATE_PROMPT, KNOWLEDGE_COMPOSE_PROMPT,
)


def test_prompts_are_nonempty_strings():
    for p in (TRIAGE_PROMPT, VERIFICATION_PROMPT, GUARDRAIL_OUTPUT_PROMPT,
              KNOWLEDGE_GRADER_PROMPT, KNOWLEDGE_REFORMULATE_PROMPT,
              KNOWLEDGE_COMPOSE_PROMPT):
        assert isinstance(p, str) and len(p) > 20


def test_triage_mentions_clarify_and_route():
    assert "clarify" in TRIAGE_PROMPT.lower()
    assert "route" in TRIAGE_PROMPT.lower()


def test_grader_judges_content_not_score():
    assert "answer_present" in KNOWLEDGE_GRADER_PROMPT


def test_compose_has_no_answer_and_bug_markers():
    assert "[[no_answer]]" in KNOWLEDGE_COMPOSE_PROMPT
    assert "suspected_bug" in KNOWLEDGE_COMPOSE_PROMPT

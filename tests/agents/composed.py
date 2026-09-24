"""Build a ComposedAnswer the way the tests find clearest to read.

KnowledgeAgent's control markers moved out of the prose and into
`ComposedAnswer.status`, but writing `"... [[clarify]]"` is still the most compact
way for a test to say what the composer decided. This translates the one into the
other, so the marker-era tests keep their shape while exercising the new schema.

Tests that are ABOUT the status field — the hedge rule, a marker contradicting the
field — build ComposedAnswer directly instead, or they would be testing this helper.
"""

from agent_customer_support.agents.knowledge import parse_markers
from agent_customer_support.llm.schemas import CitedSource, ComposedAnswer


def composed_answer(answer: str, cited: list[CitedSource] | None = None) -> ComposedAnswer:
    clean, kind, application = parse_markers(answer)
    return ComposedAnswer(
        answer=clean,
        status=kind or "answer",
        application=application or "",
        cited=cited or [],
    )

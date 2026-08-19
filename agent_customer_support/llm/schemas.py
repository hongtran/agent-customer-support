"""Response schemas for the agents that ask the LLM for a decision, not prose.

These live in the LLM layer rather than `models.py` because they describe wire
format -- what one provider call is constrained to return -- not domain state.

A schema guarantees the *shape* of an answer, never its *correctness* and never
that a call succeeded at all: both provider `parse` APIs return an Optional. Every
caller keeps its fail-safe default for the None case.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TriageDecision(BaseModel):
    """Routing target chosen by TriageAgent.

    `flow` is deliberately absent: FlowAgent is reached from the
    `session.active_flow_id` fast path in TriageAgent.run, before any LLM call, so
    the model is never the thing that picks it.
    """

    model_config = ConfigDict(extra="forbid")

    target: Literal["knowledge", "escalate", "out_of_scope"] = Field(
        description=(
            "knowledge for any product question, how-to, bug report or feature "
            "request; escalate only when the user explicitly asks for a human; "
            "out_of_scope only when the question is clearly unrelated to the "
            "CenLab software (finance, weather, sports, food, translation, "
            "general programming...) — when in doubt, pick knowledge."
        )
    )


class GuardrailVerdict(BaseModel):
    """Output-guardrail decision on a composed reply."""

    model_config = ConfigDict(extra="forbid")

    flag: bool = Field(description="true when the reply must be replaced")
    # Required, not optional: OpenAI strict mode demands every property appear in
    # `required`, and a field with a default would be dropped from it.
    reason: str = Field(description="short reason; empty string when flag is false")

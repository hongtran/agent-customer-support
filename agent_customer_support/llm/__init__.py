from functools import lru_cache
from typing import TypeVar

from pydantic import BaseModel

from agent_customer_support.config import get_settings
from agent_customer_support.llm import usage
from agent_customer_support.llm.providers.anthropic_provider import (
    anthropic_complete_with_tools,
)
from agent_customer_support.llm.providers.openai_provider import (
    openai_complete_with_tools,
)
from agent_customer_support.observability import tracing


@lru_cache
def _anthropic_client():
    from anthropic import Anthropic

    return Anthropic()  # reads ANTHROPIC_API_KEY from env


@lru_cache
def _openai_client():
    from openai import OpenAI

    return OpenAI()  # reads OPENAI_API_KEY from env


def _is_anthropic(model: str) -> bool:
    return "claude" in model


SchemaT = TypeVar("SchemaT", bound=BaseModel)


def complete_with_tools(
    *,
    messages: list[dict],
    tools: list[dict],
    system: str | list[dict] | None = None,
    model: str | None = None,
    schema: type[BaseModel] | None = None,
) -> dict:
    """One LLM call, traced and metered.

    `schema` opts into provider-side constrained decoding: the reply is guaranteed
    to validate against that Pydantic model, and the validated instance comes back
    under `"parsed"` (None if the model produced no valid instance). Leaving it None
    is the plain free-text path every other caller uses.
    """
    cfg = get_settings()
    model = model or cfg.agent_model
    with tracing.generation(
        "llm",
        model=model,
        input=messages,
        metadata={"environment": cfg.environment, "reasoning_effort": cfg.reasoning_effort},
    ) as gen:
        if _is_anthropic(model):
            out = anthropic_complete_with_tools(
                client=_anthropic_client(),
                model=model,
                messages=messages,
                tools=tools,
                system=system,
                schema=schema,
            )
        else:
            out = openai_complete_with_tools(
                client=_openai_client(),
                model=model,
                messages=messages,
                tools=tools,
                system=system,
                max_tokens=cfg.max_output_tokens,
                reasoning_effort=cfg.reasoning_effort,
                schema=schema,
            )
        gen.update(output=out.get("text"), usage_details=out.get("usage"))
        # Same numbers as the Langfuse update above, made available in-process so the
        # eval harness can price a run without a Langfuse deployment. No-op unless a
        # caller has opened a `usage.collect()` scope.
        usage.record(model, out.get("usage"))
        return out


def complete_text(
    messages: list[dict],
    system: str | list[dict] | None = None,
    model: str | None = None,
) -> str:
    out = complete_with_tools(messages=messages, tools=[], system=system, model=model)
    return out.get("text") or ""


def complete_structured(
    *,
    messages: list[dict],
    schema: type[SchemaT],
    system: str | list[dict] | None = None,
    model: str | None = None,
) -> SchemaT | None:
    """Return a validated `schema` instance, or None when the model produced none.

    None is not exceptional -- an API error, a refusal, or a truncated response all
    land here -- so every caller MUST keep a fail-safe default for it. Constrained
    decoding guarantees the *shape* of a decision, never that a decision was made
    and never that it is the right one.
    """
    out = complete_with_tools(
        messages=messages, tools=[], system=system, model=model, schema=schema
    )
    return out.get("parsed")

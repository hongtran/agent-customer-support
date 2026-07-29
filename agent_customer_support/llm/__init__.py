from functools import lru_cache

from agent_customer_support.config import get_settings
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


def complete_with_tools(
    *,
    messages: list[dict],
    tools: list[dict],
    system: str | list[dict] | None = None,
    model: str | None = None,
) -> dict:
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
            )
        gen.update(output=out.get("text"), usage_details=out.get("usage"))
        return out


def complete_text(
    messages: list[dict],
    system: str | list[dict] | None = None,
    model: str | None = None,
) -> str:
    out = complete_with_tools(messages=messages, tools=[], system=system, model=model)
    return out.get("text") or ""

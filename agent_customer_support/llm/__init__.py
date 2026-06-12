from functools import lru_cache

from agent_customer_support.config import get_settings
from agent_customer_support.llm.providers.anthropic_provider import (
    anthropic_complete_with_tools,
)
from agent_customer_support.llm.providers.openai_provider import (
    openai_complete_with_tools,
)


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
    *, messages: list[dict], tools: list[dict], system: str | None = None
) -> dict:
    model = get_settings().agent_model
    if _is_anthropic(model):
        return anthropic_complete_with_tools(
            client=_anthropic_client(), model=model,
            messages=messages, tools=tools, system=system,
        )
    return openai_complete_with_tools(
        client=_openai_client(), model=model,
        messages=messages, tools=tools, system=system,
    )


def complete_text(messages: list[dict], system: str | None = None) -> str:
    out = complete_with_tools(messages=messages, tools=[], system=system)
    return out.get("text") or ""

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

_OPENROUTER_PREFIX = "openrouter/"


@lru_cache
def _openrouter_client():
    from openai import OpenAI

    cfg = get_settings()
    if not cfg.openrouter_api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is unset but a model is routed to OpenRouter "
            "(model name starts with 'openrouter/')"
        )
    return OpenAI(base_url=cfg.openrouter_base_url, api_key=cfg.openrouter_api_key)


def _is_openrouter(model: str) -> bool:
    return model.startswith(_OPENROUTER_PREFIX)


# Self-hosted vLLM on Modal (qwen/serve.py). The rest of the name is vLLM's
# --served-model-name, e.g. `modal/qwen3.8-27b`.
_MODAL_PREFIX = "modal/"

# Qwen's chat template raises on any effort other than these three -- including
# OpenAI's "high" -- so the env-enforced profile is translated, never passed through.
_QWEN_REASONING_EFFORT = {"minimal": "low", "low": "low", "medium": "medium", "high": "xhigh"}


@lru_cache
def _modal_client():
    from openai import OpenAI

    cfg = get_settings()
    if not (cfg.modal_llm_base_url and cfg.modal_llm_api_key):
        raise RuntimeError(
            "MODAL_LLM_BASE_URL and MODAL_LLM_API_KEY must both be set when a model "
            "is routed to Modal (model name starts with 'modal/')"
        )
    return OpenAI(base_url=cfg.modal_llm_base_url, api_key=cfg.modal_llm_api_key)


def _is_modal(model: str) -> bool:
    return model.startswith(_MODAL_PREFIX)


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
        if _is_openrouter(model):
            out = openai_complete_with_tools(
                client=_openrouter_client(),
                # OpenRouter wants the bare `<vendor>/<name>`; the prefix is ours.
                model=model.removeprefix(_OPENROUTER_PREFIX),
                messages=messages,
                tools=tools,
                system=system,
                max_tokens=cfg.max_output_tokens,
                reasoning_effort=cfg.reasoning_effort,
                schema=schema,
                # OpenRouter silently DROPS params an upstream provider does not
                # support. For a schema call that means unconstrained text coming
                # back through `.parse` — a parse failure, not an error. This makes
                # it route only to providers that honour json_schema, so the failure
                # is a loud 404 the caller's fallback can act on.
                extra_body={"provider": {"require_parameters": True}} if schema else None,
            )
        elif _is_modal(model):
            out = openai_complete_with_tools(
                client=_modal_client(),
                model=model.removeprefix(_MODAL_PREFIX),
                messages=messages,
                tools=tools,
                system=system,
                # Counts thinking + answer; a truncated thinking phase leaves content
                # empty, which complete_structured reports as None.
                max_tokens=cfg.max_output_tokens,
                # Top-level reasoning_effort is an OpenAI API field; Qwen takes it
                # through the chat template instead.
                reasoning_effort=None,
                schema=schema,
                extra_body={
                    "chat_template_kwargs": {
                        "reasoning_effort": _QWEN_REASONING_EFFORT[cfg.reasoning_effort]
                    }
                },
                # Use the model's own sampling defaults (temp 1.0, top_p 0.95, top_k 20).
                temperature=None,
            )
        elif _is_anthropic(model):
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

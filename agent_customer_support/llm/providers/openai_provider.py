import json

# Reasoning-family models: they take `max_completion_tokens` + `reasoning_effort`
# and reject any `temperature` other than the default.
_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")

_DEFAULT_MAX_TOKENS = 5000
_DEFAULT_REASONING_MAX_TOKENS = 8000


def _is_reasoning_model(model: str) -> bool:
    return model.startswith(_REASONING_MODEL_PREFIXES)


def to_openai_tools(tool_defs: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object"}),
            },
        }
        for t in tool_defs
    ]


def openai_complete_with_tools(
    *,
    client,
    model: str,
    messages: list[dict],
    tools: list[dict],
    system: str | list[dict] | None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> dict:
    msgs = list(messages)
    if system:
        # Anthropic-style block lists (used for prompt caching) carry no meaning to
        # OpenAI — flatten to plain text. The concatenation order is preserved, so the
        # static prefix stays stable and OpenAI's automatic prefix caching still applies.
        if isinstance(system, list):
            system = "\n\n".join(b["text"] for b in system if b.get("type") == "text")
        msgs = [{"role": "system", "content": system}, *msgs]

    kwargs: dict = {"model": model, "messages": msgs}
    if _is_reasoning_model(model):
        # The cap counts reasoning tokens too, and temperature is not accepted.
        kwargs["max_completion_tokens"] = max_tokens or _DEFAULT_REASONING_MAX_TOKENS
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
    else:
        kwargs["max_tokens"] = max_tokens or _DEFAULT_MAX_TOKENS
        kwargs["temperature"] = 0.5
    if tools:
        kwargs["tools"] = to_openai_tools(tools)

    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    msg = choice.message

    tool_calls: list[dict] = []
    for tc in msg.tool_calls or []:
        tool_calls.append(
            {
                "id": tc.id,
                "name": tc.function.name,
                "input": json.loads(tc.function.arguments or "{}"),
            }
        )

    stop_reason = "tool_use" if tool_calls else choice.finish_reason
    usage = getattr(resp, "usage", None)
    usage_details = (
        {"input": usage.prompt_tokens, "output": usage.completion_tokens}
        if usage is not None
        else None
    )
    return {
        "stop_reason": stop_reason,
        "text": msg.content,
        "tool_calls": tool_calls,
        "usage": usage_details,
    }

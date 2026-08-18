from pydantic import BaseModel


def anthropic_complete_with_tools(
    *,
    client,
    model: str,
    messages: list[dict],
    tools: list[dict],
    system: str | list[dict] | None,
    max_tokens: int = 1500,
    schema: type[BaseModel] | None = None,
) -> dict:
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if tools:
        kwargs["tools"] = tools
    if system:
        kwargs["system"] = system

    if schema is not None:
        # `.parse` constrains decoding to the schema and validates the result into a
        # model instance. ParsedMessage subclasses Message, so the block/usage
        # extraction below is unchanged; `parsed_output` is None when the model
        # produced no valid instance.
        resp = client.messages.parse(**kwargs, output_format=schema)
        parsed = resp.parsed_output
    else:
        resp = client.messages.create(**kwargs)
        parsed = None

    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in resp.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append({"id": block.id, "name": block.name, "input": block.input})

    usage = getattr(resp, "usage", None)
    usage_details = (
        {"input": usage.input_tokens, "output": usage.output_tokens} if usage is not None else None
    )
    return {
        "stop_reason": resp.stop_reason,
        "text": "".join(text_parts) or None,
        "tool_calls": tool_calls,
        "usage": usage_details,
        "parsed": parsed,
    }

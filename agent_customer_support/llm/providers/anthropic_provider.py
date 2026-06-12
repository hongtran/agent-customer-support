def anthropic_complete_with_tools(
    *, client, model: str, messages: list[dict],
    tools: list[dict], system: str | None, max_tokens: int = 1500,
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

    resp = client.messages.create(**kwargs)

    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in resp.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append({"id": block.id, "name": block.name, "input": block.input})

    return {
        "stop_reason": resp.stop_reason,
        "text": "".join(text_parts) or None,
        "tool_calls": tool_calls,
    }

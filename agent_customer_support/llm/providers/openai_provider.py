import json


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
    *, client, model: str, messages: list[dict],
    tools: list[dict], system: str | None, max_tokens: int = 1500,
) -> dict:
    msgs = list(messages)
    if system:
        msgs = [{"role": "system", "content": system}, *msgs]

    kwargs: dict = {"model": model, "messages": msgs, "max_tokens": max_tokens}
    if tools:
        kwargs["tools"] = to_openai_tools(tools)

    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    msg = choice.message

    tool_calls: list[dict] = []
    for tc in (msg.tool_calls or []):
        tool_calls.append({
            "id": tc.id,
            "name": tc.function.name,
            "input": json.loads(tc.function.arguments or "{}"),
        })

    stop_reason = "tool_use" if tool_calls else choice.finish_reason
    return {
        "stop_reason": stop_reason,
        "text": msg.content,
        "tool_calls": tool_calls,
    }

import json
from types import SimpleNamespace
from unittest.mock import MagicMock
from agent_customer_support.llm.providers.openai_provider import (
    openai_complete_with_tools, to_openai_tools,
)


def test_to_openai_tools_shape():
    defs = [{"name": "f", "description": "d", "input_schema": {"type": "object"}}]
    out = to_openai_tools(defs)
    assert out == [{
        "type": "function",
        "function": {"name": "f", "description": "d", "parameters": {"type": "object"}},
    }]


def test_parses_text_response():
    msg = SimpleNamespace(content="hello", tool_calls=None)
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=9, completion_tokens=5),
    )
    client = MagicMock()
    client.chat.completions.create.return_value = resp
    out = openai_complete_with_tools(
        client=client, model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}], tools=[], system="sys",
    )
    assert out["stop_reason"] == "stop"
    assert out["text"] == "hello"
    assert out["tool_calls"] == []


def test_parses_tool_calls():
    tc = SimpleNamespace(
        id="t1",
        function=SimpleNamespace(name="search_knowledge",
                                 arguments=json.dumps({"query": "x"})),
    )
    msg = SimpleNamespace(content=None, tool_calls=[tc])
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")],
        usage=SimpleNamespace(prompt_tokens=9, completion_tokens=5),
    )
    client = MagicMock()
    client.chat.completions.create.return_value = resp
    out = openai_complete_with_tools(
        client=client, model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"name": "search_knowledge", "description": "d", "input_schema": {}}],
        system=None,
    )
    assert out["stop_reason"] == "tool_use"
    assert out["tool_calls"] == [
        {"id": "t1", "name": "search_knowledge", "input": {"query": "x"}}
    ]


def test_surfaces_usage():
    msg = SimpleNamespace(content="hello", tool_calls=None)
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=9, completion_tokens=5),
    )
    client = MagicMock()
    client.chat.completions.create.return_value = resp
    out = openai_complete_with_tools(
        client=client, model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}], tools=[], system=None,
    )
    assert out["usage"] == {"input": 9, "output": 5}


def test_usage_none_safe():
    msg = SimpleNamespace(content="hello", tool_calls=None)
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=None,
    )
    client = MagicMock()
    client.chat.completions.create.return_value = resp
    out = openai_complete_with_tools(
        client=client, model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}], tools=[], system=None,
    )
    assert out["usage"] is None

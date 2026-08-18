import json
from types import SimpleNamespace
from unittest.mock import MagicMock
from agent_customer_support.llm.schemas import TriageDecision
from agent_customer_support.llm.providers.openai_provider import (
    openai_complete_with_tools,
    to_openai_tools,
)


def test_to_openai_tools_shape():
    defs = [{"name": "f", "description": "d", "input_schema": {"type": "object"}}]
    out = to_openai_tools(defs)
    assert out == [
        {
            "type": "function",
            "function": {"name": "f", "description": "d", "parameters": {"type": "object"}},
        }
    ]


def test_parses_text_response():
    msg = SimpleNamespace(content="hello", tool_calls=None)
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=9, completion_tokens=5),
    )
    client = MagicMock()
    client.chat.completions.create.return_value = resp
    out = openai_complete_with_tools(
        client=client,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        system="sys",
    )
    assert out["stop_reason"] == "stop"
    assert out["text"] == "hello"
    assert out["tool_calls"] == []


def test_parses_tool_calls():
    tc = SimpleNamespace(
        id="t1",
        function=SimpleNamespace(name="search_knowledge", arguments=json.dumps({"query": "x"})),
    )
    msg = SimpleNamespace(content=None, tool_calls=[tc])
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")],
        usage=SimpleNamespace(prompt_tokens=9, completion_tokens=5),
    )
    client = MagicMock()
    client.chat.completions.create.return_value = resp
    out = openai_complete_with_tools(
        client=client,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"name": "search_knowledge", "description": "d", "input_schema": {}}],
        system=None,
    )
    assert out["stop_reason"] == "tool_use"
    assert out["tool_calls"] == [{"id": "t1", "name": "search_knowledge", "input": {"query": "x"}}]


def test_surfaces_usage():
    msg = SimpleNamespace(content="hello", tool_calls=None)
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=9, completion_tokens=5),
    )
    client = MagicMock()
    client.chat.completions.create.return_value = resp
    out = openai_complete_with_tools(
        client=client,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        system=None,
    )
    assert out["usage"] == {"input": 9, "output": 5}


def _text_client():
    msg = SimpleNamespace(content="hello", tool_calls=None)
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=9, completion_tokens=5),
    )
    client = MagicMock()
    client.chat.completions.create.return_value = resp
    return client


def test_reasoning_model_gets_reasoning_params():
    client = _text_client()
    openai_complete_with_tools(
        client=client,
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        system=None,
        max_tokens=8000,
        reasoning_effort="high",
    )
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["max_completion_tokens"] == 8000
    assert kwargs["reasoning_effort"] == "high"
    # reasoning models reject any temperature other than the default
    assert "temperature" not in kwargs
    assert "max_tokens" not in kwargs


def test_legacy_model_keeps_chat_params():
    client = _text_client()
    openai_complete_with_tools(
        client=client,
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        system=None,
        max_tokens=8000,
        reasoning_effort="high",
    )
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["max_tokens"] == 8000
    assert kwargs["temperature"] == 0.5
    assert "reasoning_effort" not in kwargs
    assert "max_completion_tokens" not in kwargs


def test_usage_none_safe():
    msg = SimpleNamespace(content="hello", tool_calls=None)
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        usage=None,
    )
    client = MagicMock()
    client.chat.completions.create.return_value = resp
    out = openai_complete_with_tools(
        client=client,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        system=None,
    )
    assert out["usage"] is None


def test_schema_uses_parse_and_returns_instance():
    """With a schema, the call must go to `.parse` (constrained decoding), not
    `.create`, and the validated instance rides back under "parsed"."""
    decision = TriageDecision(target="escalate")
    msg = SimpleNamespace(
        content='{"target":"escalate"}', tool_calls=None, parsed=decision, refusal=None
    )
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=9, completion_tokens=5),
    )
    client = MagicMock()
    client.chat.completions.parse.return_value = resp
    out = openai_complete_with_tools(
        client=client,
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        system="sys",
        schema=TriageDecision,
    )
    client.chat.completions.parse.assert_called_once()
    client.chat.completions.create.assert_not_called()
    assert client.chat.completions.parse.call_args.kwargs["response_format"] is TriageDecision
    assert out["parsed"] is decision


def test_no_schema_still_uses_create_and_parsed_is_none():
    msg = SimpleNamespace(content="hello", tool_calls=None)
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=9, completion_tokens=5),
    )
    client = MagicMock()
    client.chat.completions.create.return_value = resp
    out = openai_complete_with_tools(
        client=client,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        system="sys",
    )
    client.chat.completions.create.assert_called_once()
    client.chat.completions.parse.assert_not_called()
    assert out["parsed"] is None


def test_refusal_yields_no_parsed_instance():
    """A refusal is a real failure for the caller's fallback, not a parse bug —
    and never a half-valid instance handed on as if the model had decided."""
    msg = SimpleNamespace(
        content=None, tool_calls=None, parsed=TriageDecision(target="knowledge"), refusal="no"
    )
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=9, completion_tokens=5),
    )
    client = MagicMock()
    client.chat.completions.parse.return_value = resp
    out = openai_complete_with_tools(
        client=client,
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        system="sys",
        schema=TriageDecision,
    )
    assert out["parsed"] is None

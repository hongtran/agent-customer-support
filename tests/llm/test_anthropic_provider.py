from types import SimpleNamespace
from unittest.mock import MagicMock
from agent_customer_support.llm.providers.anthropic_provider import (
    anthropic_complete_with_tools,
)


def _block(**kw):
    return SimpleNamespace(**kw)


def test_parses_text_response():
    resp = SimpleNamespace(
        stop_reason="end_turn",
        content=[_block(type="text", text="hello")],
    )
    client = MagicMock()
    client.messages.create.return_value = resp
    out = anthropic_complete_with_tools(
        client=client, model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "hi"}],
        tools=[], system="sys",
    )
    assert out["stop_reason"] == "end_turn"
    assert out["text"] == "hello"
    assert out["tool_calls"] == []


def test_parses_tool_use():
    resp = SimpleNamespace(
        stop_reason="tool_use",
        content=[
            _block(type="text", text="let me check"),
            _block(type="tool_use", id="t1", name="search_knowledge",
                   input={"query": "x"}),
        ],
    )
    client = MagicMock()
    client.messages.create.return_value = resp
    out = anthropic_complete_with_tools(
        client=client, model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"name": "search_knowledge"}], system=None,
    )
    assert out["stop_reason"] == "tool_use"
    assert out["text"] == "let me check"
    assert out["tool_calls"] == [
        {"id": "t1", "name": "search_knowledge", "input": {"query": "x"}}
    ]

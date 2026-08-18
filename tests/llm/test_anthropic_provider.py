from types import SimpleNamespace
from unittest.mock import MagicMock
from agent_customer_support.llm.schemas import TriageDecision
from agent_customer_support.llm.providers.anthropic_provider import (
    anthropic_complete_with_tools,
)


def _block(**kw):
    return SimpleNamespace(**kw)


def test_parses_text_response():
    resp = SimpleNamespace(
        stop_reason="end_turn",
        content=[_block(type="text", text="hello")],
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
    )
    client = MagicMock()
    client.messages.create.return_value = resp
    out = anthropic_complete_with_tools(
        client=client,
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        system="sys",
    )
    assert out["stop_reason"] == "end_turn"
    assert out["text"] == "hello"
    assert out["tool_calls"] == []


def test_parses_tool_use():
    resp = SimpleNamespace(
        stop_reason="tool_use",
        content=[
            _block(type="text", text="let me check"),
            _block(type="tool_use", id="t1", name="search_knowledge", input={"query": "x"}),
        ],
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
    )
    client = MagicMock()
    client.messages.create.return_value = resp
    out = anthropic_complete_with_tools(
        client=client,
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"name": "search_knowledge"}],
        system=None,
    )
    assert out["stop_reason"] == "tool_use"
    assert out["text"] == "let me check"
    assert out["tool_calls"] == [{"id": "t1", "name": "search_knowledge", "input": {"query": "x"}}]


def test_surfaces_usage():
    resp = SimpleNamespace(
        stop_reason="end_turn",
        content=[_block(type="text", text="hi")],
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
    )
    client = MagicMock()
    client.messages.create.return_value = resp
    out = anthropic_complete_with_tools(
        client=client,
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        system=None,
    )
    assert out["usage"] == {"input": 11, "output": 7}


def test_schema_uses_parse_and_returns_instance():
    """The Anthropic path must get real constrained decoding too — `.parse` with
    `output_format`, not `.create`. ParsedMessage subclasses Message, so the block
    and usage extraction is unchanged."""
    decision = TriageDecision(target="escalate")
    resp = SimpleNamespace(
        stop_reason="end_turn",
        content=[_block(type="text", text='{"target":"escalate"}')],
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
        parsed_output=decision,
    )
    client = MagicMock()
    client.messages.parse.return_value = resp
    out = anthropic_complete_with_tools(
        client=client,
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        system="sys",
        schema=TriageDecision,
    )
    client.messages.parse.assert_called_once()
    client.messages.create.assert_not_called()
    assert client.messages.parse.call_args.kwargs["output_format"] is TriageDecision
    assert out["parsed"] is decision
    assert out["text"] == '{"target":"escalate"}'
    assert out["usage"] == {"input": 11, "output": 7}


def test_no_schema_still_uses_create_and_parsed_is_none():
    resp = SimpleNamespace(
        stop_reason="end_turn",
        content=[_block(type="text", text="hello")],
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
    )
    client = MagicMock()
    client.messages.create.return_value = resp
    out = anthropic_complete_with_tools(
        client=client,
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        system="sys",
    )
    client.messages.create.assert_called_once()
    client.messages.parse.assert_not_called()
    assert out["parsed"] is None


def test_schema_parse_returns_none_when_no_valid_instance():
    resp = SimpleNamespace(
        stop_reason="max_tokens",
        content=[_block(type="text", text='{"targ')],
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
        parsed_output=None,
    )
    client = MagicMock()
    client.messages.parse.return_value = resp
    out = anthropic_complete_with_tools(
        client=client,
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        system="sys",
        schema=TriageDecision,
    )
    assert out["parsed"] is None

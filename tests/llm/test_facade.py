from unittest.mock import patch, MagicMock
from agent_customer_support.llm import complete_with_tools


def test_routes_to_anthropic_for_claude_model():
    fake = {"stop_reason": "end_turn", "text": "hi", "tool_calls": []}
    with (
        patch("agent_customer_support.llm.get_settings") as gs,
        patch("agent_customer_support.llm._anthropic_client", return_value=MagicMock()),
        patch("agent_customer_support.llm.anthropic_complete_with_tools", return_value=fake) as m,
    ):
        gs.return_value.agent_model = "claude-3-5-sonnet"
        out = complete_with_tools(
            messages=[{"role": "user", "content": "x"}], tools=[], system=None
        )
    assert out["text"] == "hi"
    m.assert_called_once()
    # reasoning effort is an OpenAI-side concept — never forwarded to Anthropic
    assert "reasoning_effort" not in m.call_args.kwargs
    assert "max_tokens" not in m.call_args.kwargs


def test_routes_to_openai_for_gpt_model():
    fake = {"stop_reason": "stop", "text": "hi", "tool_calls": []}
    with (
        patch("agent_customer_support.llm.get_settings") as gs,
        patch("agent_customer_support.llm._openai_client", return_value=MagicMock()),
        patch("agent_customer_support.llm.openai_complete_with_tools", return_value=fake) as m,
    ):
        gs.return_value.agent_model = "gpt-4o-mini"
        out = complete_with_tools(
            messages=[{"role": "user", "content": "x"}], tools=[], system=None
        )
    assert out["text"] == "hi"
    m.assert_called_once()


def test_forwards_settings_reasoning_profile_to_openai():
    fake = {"stop_reason": "stop", "text": "hi", "tool_calls": []}
    with (
        patch("agent_customer_support.llm.get_settings") as gs,
        patch("agent_customer_support.llm._openai_client", return_value=MagicMock()),
        patch("agent_customer_support.llm.openai_complete_with_tools", return_value=fake) as m,
    ):
        gs.return_value.agent_model = "gpt-5.4-mini"
        gs.return_value.reasoning_effort = "high"
        gs.return_value.max_output_tokens = 8000
        complete_with_tools(messages=[{"role": "user", "content": "x"}], tools=[], system=None)
    assert m.call_args.kwargs["reasoning_effort"] == "high"
    assert m.call_args.kwargs["max_tokens"] == 8000

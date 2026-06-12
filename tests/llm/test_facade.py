from unittest.mock import patch, MagicMock
from agent_customer_support.llm import complete_with_tools, complete_text


def test_routes_to_anthropic_for_claude_model():
    fake = {"stop_reason": "end_turn", "text": "hi", "tool_calls": []}
    with patch("agent_customer_support.llm.get_settings") as gs, \
         patch("agent_customer_support.llm._anthropic_client", return_value=MagicMock()), \
         patch("agent_customer_support.llm.anthropic_complete_with_tools",
               return_value=fake) as m:
        gs.return_value.agent_model = "claude-3-5-sonnet"
        out = complete_with_tools(messages=[{"role": "user", "content": "x"}],
                                  tools=[], system=None)
    assert out["text"] == "hi"
    m.assert_called_once()


def test_routes_to_openai_for_gpt_model():
    fake = {"stop_reason": "stop", "text": "hi", "tool_calls": []}
    with patch("agent_customer_support.llm.get_settings") as gs, \
         patch("agent_customer_support.llm._openai_client", return_value=MagicMock()), \
         patch("agent_customer_support.llm.openai_complete_with_tools",
               return_value=fake) as m:
        gs.return_value.agent_model = "gpt-4o-mini"
        out = complete_with_tools(messages=[{"role": "user", "content": "x"}],
                                  tools=[], system=None)
    assert out["text"] == "hi"
    m.assert_called_once()

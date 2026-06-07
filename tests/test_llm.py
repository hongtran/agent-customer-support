from unittest.mock import patch
from agent_customer_support.llm import complete_with_tools, complete_text


def test_complete_with_tools_delegates():
    with patch("agent_customer_support.llm.ai_completion_with_tools") as f:
        f.return_value = {"stop_reason": "end", "text": "hi", "tool_calls": [], "raw": None}
        out = complete_with_tools(messages=[{"role": "user", "content": "x"}], tools=[], system="s")
    assert out["text"] == "hi"
    f.assert_called_once()


def test_complete_text_delegates():
    with patch("agent_customer_support.llm.ai_completion") as f:
        f.return_value = {"content": "answer"}
        out = complete_text([{"role": "user", "content": "x"}])
    assert out == "answer"

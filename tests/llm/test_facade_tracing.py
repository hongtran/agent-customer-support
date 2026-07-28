from contextlib import contextmanager
from unittest.mock import MagicMock, patch
from agent_customer_support.llm import complete_with_tools


def test_facade_records_generation_with_usage():
    handle = MagicMock()

    @contextmanager
    def fake_gen(name, *, model, input=None, metadata=None):
        yield handle

    fake_out = {
        "stop_reason": "end_turn",
        "text": "hi",
        "tool_calls": [],
        "usage": {"input": 3, "output": 2},
    }
    with (
        patch("agent_customer_support.llm.tracing.generation", fake_gen),
        patch("agent_customer_support.llm.get_settings") as gs,
        patch("agent_customer_support.llm._openai_client", return_value=MagicMock()),
        patch("agent_customer_support.llm.openai_complete_with_tools", return_value=fake_out),
    ):
        gs.return_value.agent_model = "gpt-4o-mini"
        out = complete_with_tools(
            messages=[{"role": "user", "content": "x"}], tools=[], system=None
        )
    assert out["text"] == "hi"
    handle.update.assert_called_once_with(output="hi", usage_details={"input": 3, "output": 2})

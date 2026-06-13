from unittest.mock import MagicMock
import pytest
from agent_customer_support.observability import tracing


def test_disabled_when_no_client(monkeypatch):
    monkeypatch.setattr(tracing, "_client", lambda: None)
    assert tracing.enabled() is False


def test_span_noop_when_disabled_runs_body(monkeypatch):
    monkeypatch.setattr(tracing, "_client", lambda: None)
    ran = {}
    with tracing.span("x", input={"a": 1}) as s:
        ran["yes"] = True
        s.update(output="ok")  # must not raise
    assert ran["yes"] is True


def test_generation_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(tracing, "_client", lambda: None)
    with tracing.generation("llm", model="gpt-4o-mini") as g:
        g.update(output="hi", usage_details={"input": 1, "output": 2})  # no raise


def test_trace_noop_when_disabled_runs_body(monkeypatch):
    monkeypatch.setattr(tracing, "_client", lambda: None)
    ran = {}
    with tracing.trace("turn", session_id="cv1", user_id="c1") as t:
        ran["yes"] = True
        t.update(output="reply")  # no raise
    assert ran["yes"] is True
    tracing.flush()  # no raise


def test_span_uses_observation_api_when_enabled(monkeypatch):
    fake_client = MagicMock()
    fake_cm = MagicMock()
    fake_handle = MagicMock()
    fake_cm.__enter__.return_value = fake_handle
    fake_cm.__exit__.return_value = False
    fake_client.start_as_current_observation.return_value = fake_cm
    monkeypatch.setattr(tracing, "_client", lambda: fake_client)
    with tracing.span("agent.triage", input={"m": "hi"}) as s:
        assert s is fake_handle
    fake_client.start_as_current_observation.assert_called_once()
    assert fake_client.start_as_current_observation.call_args.kwargs["as_type"] == "span"


def test_generation_uses_generation_type(monkeypatch):
    fake_client = MagicMock()
    fake_cm = MagicMock()
    fake_cm.__enter__.return_value = MagicMock()
    fake_cm.__exit__.return_value = False
    fake_client.start_as_current_observation.return_value = fake_cm
    monkeypatch.setattr(tracing, "_client", lambda: fake_client)
    with tracing.generation("llm", model="gpt-4o-mini"):
        pass
    kwargs = fake_client.start_as_current_observation.call_args.kwargs
    assert kwargs["as_type"] == "generation"
    assert kwargs["model"] == "gpt-4o-mini"


def test_body_exception_propagates_when_enabled(monkeypatch):
    fake_client = MagicMock()
    fake_cm = MagicMock()
    fake_cm.__enter__.return_value = MagicMock()
    fake_cm.__exit__.return_value = False
    fake_client.start_as_current_observation.return_value = fake_cm
    monkeypatch.setattr(tracing, "_client", lambda: fake_client)
    with pytest.raises(ValueError):
        with tracing.span("x"):
            raise ValueError("boom")


def test_start_failure_falls_back_to_noop(monkeypatch):
    fake_client = MagicMock()
    fake_client.start_as_current_observation.side_effect = RuntimeError("sdk error")
    monkeypatch.setattr(tracing, "_client", lambda: fake_client)
    with tracing.span("x") as s:
        s.update(output="ok")  # must not raise; body still runs

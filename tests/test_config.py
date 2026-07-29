import pytest
from pydantic import ValidationError

from agent_customer_support.config import Settings, get_settings


def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("PRODUCT_COLLECTION", "cenlab")
    monkeypatch.setenv("AGENT_MODEL", "gpt-4o-mini")
    get_settings.cache_clear()
    s = get_settings()
    assert s.product_collection == "cenlab"
    assert s.agent_model == "gpt-4o-mini"
    assert s.session_ttl_seconds == 3600


# These build Settings() directly rather than through the lru_cached get_settings(),
# so they cannot leave a stale cached instance behind for the rest of the suite.
def test_dev_environment_uses_low_effort(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "dev")
    s = Settings()
    assert s.environment == "dev"
    assert s.reasoning_effort == "low"
    assert s.max_output_tokens == 4000


def test_prod_environment_uses_high_effort(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "prod")
    s = Settings()
    assert s.environment == "prod"
    assert s.reasoning_effort == "high"
    assert s.max_output_tokens == 8000


def test_unknown_environment_rejected():
    with pytest.raises(ValidationError):
        Settings(environment="production")


def test_model_for_per_agent_defaults(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "gpt-5.4-mini")
    s = Settings()
    assert s.model_for("knowledge") == "gpt-5.6-luna"
    assert s.model_for("triage") == "gpt-5.4-mini"
    assert s.model_for("guardrail") == "gpt-5.4-mini"


def test_model_for_falls_back_to_agent_model_for_unknown_agent(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "claude-sonnet-4-6")
    s = Settings()
    assert s.model_for("escalation") == "claude-sonnet-4-6"

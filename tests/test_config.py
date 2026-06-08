from agent_customer_support.config import get_settings


def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("PRODUCT_COLLECTION", "cenlab")
    monkeypatch.setenv("AGENT_MODEL", "gpt-4o-mini")
    get_settings.cache_clear()
    s = get_settings()
    assert s.product_collection == "cenlab"
    assert s.agent_model == "gpt-4o-mini"
    assert s.session_ttl_seconds == 3600

from agent_customer_support.config import Settings


def test_rag_settings_have_expected_defaults():
    s = Settings()
    assert s.qdrant_endpoint  # from QDRANT_ENDPOINT env stub
    assert s.embedding_model == "gemini-embedding-001"
    assert s.embedding_dim == 3072
    assert hasattr(s, "google_api_key")

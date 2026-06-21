import pytest
import agent_customer_support.rag.embeddings as emb

pytestmark = pytest.mark.asyncio


async def test_embed_query_calls_gemini_with_retrieval_query(monkeypatch):
    captured = {}

    class FakeEmbeddings:
        def __init__(self, values):
            self.embeddings = [type("E", (), {"values": values})()]

    class FakeAio:
        class models:
            @staticmethod
            async def embed_content(*, model, contents, config):
                captured["model"] = model
                captured["contents"] = contents
                captured["task_type"] = config.task_type
                captured["dim"] = config.output_dimensionality
                return FakeEmbeddings([0.1, 0.2, 0.3])

    class FakeClient:
        aio = FakeAio()

    monkeypatch.setattr(emb, "_client", lambda: FakeClient())

    vec = await emb.embed_query("cách tạo mẫu")

    assert vec == [0.1, 0.2, 0.3]
    assert captured["model"] == "gemini-embedding-001"
    assert captured["contents"] == "cách tạo mẫu"
    assert captured["task_type"] == "RETRIEVAL_QUERY"
    assert captured["dim"] == 3072

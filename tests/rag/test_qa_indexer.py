import pytest
from qdrant_client import AsyncQdrantClient

from agent_customer_support.models import QARecord
from agent_customer_support.rag.qa_indexer import QAIndexer
import agent_customer_support.rag.qa_indexer as qa_indexer_mod
from agent_customer_support.rag_client import _normalize_collection
from agent_customer_support.config import get_settings

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _stub_embed(monkeypatch):
    captured = {}

    async def fake_embed_document(text):
        captured["text"] = text
        return [1.0, 0.0, 0.0]

    monkeypatch.setattr(qa_indexer_mod, "embed_document", fake_embed_document)
    monkeypatch.setattr(get_settings(), "embedding_dim", 3, raising=False)
    return captured


async def test_upsert_embeds_question_and_stores_answer(_stub_embed):
    client = AsyncQdrantClient(location=":memory:")
    idx = QAIndexer(client=client)
    rec = QARecord(question="đổi mật khẩu?", answer="Vào Cài đặt", source="manual",
                   application="Lab")
    await idx.upsert(rec)

    # the vector came from the QUESTION
    assert _stub_embed["text"] == "đổi mật khẩu?"

    physical = _normalize_collection(get_settings().qa_collection)
    got = await client.retrieve(physical, ids=[rec.id], with_payload=True)
    assert len(got) == 1
    payload = got[0].payload
    assert payload["page_content"] == "Vào Cài đặt"           # answer is the content
    assert payload["metadata"]["doc_type"] == "qa"
    assert payload["metadata"]["source_doc_id"] == rec.id
    assert payload["metadata"]["application"] == "Lab"


async def test_delete_removes_point(_stub_embed):
    client = AsyncQdrantClient(location=":memory:")
    idx = QAIndexer(client=client)
    rec = QARecord(question="q", answer="a", source="manual")
    await idx.upsert(rec)
    await idx.delete(rec.id)
    physical = _normalize_collection(get_settings().qa_collection)
    got = await client.retrieve(physical, ids=[rec.id], with_payload=True)
    assert got == []

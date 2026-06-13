import pytest
import respx
import httpx
from contextlib import contextmanager
from unittest.mock import MagicMock
from agent_customer_support.rag_client import RagClient
import agent_customer_support.rag_client as rag_client_mod

pytestmark = pytest.mark.asyncio


@respx.mock
async def test_search_returns_passages_and_citations():
    respx.post("http://localhost:7799/rag/query").mock(
        return_value=httpx.Response(
            200,
            json={
                "documents": ["Bước 1: vào menu X", "Bước 2: nhấn Lưu"],
                "metadatas": [
                    {"confidence": 0.82, "source_doc_id": "hdsd#3.4"},
                    {"confidence": 0.5, "source_doc_id": "hdsd#3.4"},
                ],
            },
        )
    )
    client = RagClient(base_url="http://localhost:7799")
    res = await client.search("cách tạo mẫu", collection="cenlab")
    assert res["top_confidence"] == 0.82
    assert "Bước 1" in res["passages"][0]
    assert "hdsd#3.4" in res["citations"]


@respx.mock
async def test_grounding_note_is_neutral_hint():
    respx.post("http://localhost:7799/rag/query").mock(
        return_value=httpx.Response(
            200,
            json={
                "documents": ["doc"],
                "metadatas": [{"confidence": 0.7, "source_doc_id": "hdsd#1"}],
            },
        )
    )
    client = RagClient(base_url="http://localhost:7799")
    res = await client.search("q", collection="cenlab")
    note = res["grounding_note"]
    assert "log_request" not in note
    assert "clarification" not in note
    assert "0.7" in note or "confidence" in note.lower()


@respx.mock
async def test_search_invokes_tracing_span(monkeypatch):
    respx.post("http://localhost:7799/rag/query").mock(
        return_value=httpx.Response(
            200,
            json={
                "documents": ["Bước 1"],
                "metadatas": [{"confidence": 0.75, "source_doc_id": "hdsd#5"}],
            },
        )
    )

    handle = MagicMock()
    calls: dict = {}

    @contextmanager
    def fake_span(name, *, input=None, metadata=None):
        calls["name"] = name
        yield handle

    monkeypatch.setattr(rag_client_mod.tracing, "span", fake_span)

    client = RagClient(base_url="http://localhost:7799")
    res = await client.search("cách tạo mẫu", collection="cenlab")

    assert calls["name"] == "rag.search"
    handle.update.assert_called_once()
    assert res["top_confidence"] == 0.75

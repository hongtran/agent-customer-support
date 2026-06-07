import pytest
import respx
import httpx
from agent_customer_support.rag_client import RagClient

pytestmark = pytest.mark.asyncio


@respx.mock
async def test_search_returns_passages_and_citations():
    respx.post("http://localhost:7799/rag/query").mock(
        return_value=httpx.Response(200, json={
            "documents": ["Bước 1: vào menu X", "Bước 2: nhấn Lưu"],
            "metadatas": [{"confidence": 0.82, "source_doc_id": "hdsd#3.4"},
                          {"confidence": 0.5, "source_doc_id": "hdsd#3.4"}],
        })
    )
    client = RagClient(base_url="http://localhost:7799")
    res = await client.search("cách tạo mẫu", collection="cenlab")
    assert res["top_confidence"] == 0.82
    assert "Bước 1" in res["passages"][0]
    assert "hdsd#3.4" in res["citations"]

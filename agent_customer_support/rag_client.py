import httpx
from agent_customer_support.config import get_settings


class RagClient:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or get_settings().rag_base_url

    async def search(
        self,
        query: str,
        collection: str,
        top_k: int = 8,
        score_threshold: float = 0.4,
    ) -> dict:
        payload = {
            "query": query,
            "collection_name": collection,
            "top_k": top_k,
            "score_threshold": score_threshold,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{self.base_url}/rag/query", json=payload)
            resp.raise_for_status()
            data = resp.json()
        docs = data.get("documents", []) or []
        metas = data.get("metadatas", []) or []
        confs = [m.get("confidence", 0.0) for m in metas]
        citations = sorted({
            m.get("source_doc_id") or m.get("doc_id", "")
            for m in metas if (m.get("source_doc_id") or m.get("doc_id"))
        })
        return {
            "passages": docs,
            "citations": citations,
            "top_confidence": max(confs) if confs else 0.0,
        }

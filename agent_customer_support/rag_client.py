import httpx
from agent_customer_support.config import get_settings
from agent_customer_support.observability import tracing


class RagClient:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or get_settings().rag_base_url

    async def search(
        self,
        query: str,
        collection: str,
        top_k: int = 8,
        score_threshold: float = 0.6,
        doc_type: str | None = None,
        module: str | None = None,
    ) -> dict:
        with tracing.span("rag.search",
                          input={"query": query, "collection": collection}) as sp:
            payload = {
                "query": query,
                "collection_name": collection,
                "top_k": top_k,
                "score_threshold": score_threshold,
                "doc_type": doc_type,
                "module": module,
            }
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(f"{self.base_url}/rag/query", json=payload)
                resp.raise_for_status()
                data = resp.json()
            docs = data.get("documents", []) or []
            metas = data.get("metadatas", []) or []
            confs = [m.get("confidence", 0.0) for m in metas]
            citations = sorted(
                {
                    m.get("source_doc_id") or m.get("doc_id", "")
                    for m in metas
                    if (m.get("source_doc_id") or m.get("doc_id"))
                }
            )
            top_conf = max(confs) if confs else 0.0

            # grounding_note is a HINT ONLY. The similarity score does not tell you whether
            # the passages actually answer the question — KnowledgeAgent judges that from the
            # passage text. We just surface the score and defer the decision.
            grounding = (
                f"confidence={top_conf:.2f} (chỉ là gợi ý độ tương đồng, KHÔNG phải độ đúng). "
                "Hãy tự đánh giá các passages có TRỰC TIẾP trả lời câu hỏi hay không."
            )

            result = {
                "passages": docs,
                "citations": citations,
                "top_confidence": top_conf,
                "grounding_note": grounding,
            }
            sp.update(output={"top_confidence": top_conf,
                               "n_passages": len(docs), "citations": citations})
            return result

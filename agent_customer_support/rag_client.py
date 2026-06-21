from typing import Any

from qdrant_client import AsyncQdrantClient

from agent_customer_support.config import get_settings
from agent_customer_support.observability import tracing
from agent_customer_support.rag.embeddings import embed_query

MIN_SCORE_THRESHOLD = 0.25


def _meta(point: Any) -> dict:
    """Metadata dict for a Qdrant point (langchain_qdrant nests it under 'metadata')."""
    return (point.payload or {}).get("metadata", {})


def _text(point: Any) -> str:
    return (point.payload or {}).get("page_content", "")


class RagClient:
    def __init__(self, client: AsyncQdrantClient | None = None) -> None:
        cfg = get_settings()
        self._client = client or AsyncQdrantClient(
            url=cfg.qdrant_endpoint, api_key=cfg.qdrant_api_key
        )

    async def search(
        self,
        query: str,
        collection: str,
        top_k: int = 8,
        score_threshold: float = 0.6,
        doc_type: str | None = None,
        applications: list[str] | None = None,
    ) -> dict:
        with tracing.span(
            "rag.search",
            input={"query": query, "collection": collection, "applications": applications or []},
        ) as sp:
            vec = await embed_query(query)
            resp = await self._client.query_points(
                collection_name=collection,
                query=vec,
                limit=top_k * 4,
                with_payload=True,
            )
            points = resp.points

            # Threshold; fall back to floor so callers always get the best available.
            above = [(p, p.score) for p in points if p.score >= score_threshold]
            if not above:
                above = [(p, p.score) for p in points if p.score >= MIN_SCORE_THRESHOLD]

            # doc_type AND application filter; silently relax if it removes everything.
            if doc_type or applications:
                filtered = [
                    (p, s)
                    for p, s in above
                    if (not doc_type or _meta(p).get("doc_type") == doc_type)
                    and (not applications or _meta(p).get("application") in applications)
                ]
                if filtered:
                    above = filtered

            # Deduplicate: keep highest-scoring chunk per source document.
            best: dict[str, tuple[Any, float]] = {}
            for p, s in above:
                m = _meta(p)
                sid = m.get("source_doc_id") or m.get("doc_id", "")
                if sid not in best or s > best[sid][1]:
                    best[sid] = (p, s)

            ranked = sorted(best.values(), key=lambda x: x[1], reverse=True)[:top_k]

            passages = [_text(p) for p, _ in ranked]
            metas = [{**_meta(p), "confidence": round(float(s), 4)} for p, s in ranked]
            confs = [m["confidence"] for m in metas]
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
                "passages": passages,
                "citations": citations,
                "top_confidence": top_conf,
                "grounding_note": grounding,
            }
            sp.update(
                output={
                    "top_confidence": top_conf,
                    "n_passages": len(passages),
                    "citations": citations,
                }
            )
            return result

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
        score_threshold: float = 0.6,
        doc_type: str | None = None,
        module: str | None = None,
    ) -> dict:
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

        # grounding_note: explicit hint for the LLM about how much to trust these passages.
        # High confidence (≥0.7) = passages likely contain a direct answer.
        # Medium (0.5–0.7) = related topic, verify answer is explicitly stated before using.
        # Low (<0.5) = weak match; do NOT answer from these passages.
        if top_conf >= 0.80:
            grounding = f"confidence={top_conf:.2f} — tài liệu có độ liên quan cao, kiểm tra xem có trả lời TRỰC TIẾP câu hỏi không."
        elif top_conf >= 0.50:
            grounding = (
                f"confidence={top_conf:.2f} — tài liệu liên quan nhưng yếu. "
                "Nếu đoạn trích TRẢ LỜI TRỰC TIẾP câu hỏi thì dùng. "
                "Nếu KHÔNG tìm thấy câu trả lời cụ thể → ĐẶT CÂU HỎI LÀM RÕ (clarification) "
                "để người dùng cung cấp thêm thông tin; KHÔNG gọi log_request ngay."
            )
        else:
            grounding = f"confidence={top_conf:.2f} — độ liên quan thấp. KHÔNG trả lời từ đây, hãy gọi log_request."

        return {
            "passages": docs,
            "citations": citations,
            "top_confidence": top_conf,
            "grounding_note": grounding,
        }

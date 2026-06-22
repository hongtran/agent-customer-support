from qdrant_client import AsyncQdrantClient, models

from agent_customer_support.config import get_settings
from agent_customer_support.models import QARecord
from agent_customer_support.rag.embeddings import embed_document
from agent_customer_support.rag_client import _normalize_collection


class QAIndexer:
    def __init__(self, client: AsyncQdrantClient | None = None) -> None:
        cfg = get_settings()
        self._client = client or AsyncQdrantClient(
            url=cfg.qdrant_endpoint, api_key=cfg.qdrant_api_key
        )
        self._collection = _normalize_collection(cfg.qa_collection)

    async def ensure_collection(self) -> None:
        if not await self._client.collection_exists(self._collection):
            await self._client.create_collection(
                self._collection,
                vectors_config=models.VectorParams(
                    size=get_settings().embedding_dim, distance=models.Distance.COSINE
                ),
            )

    async def upsert(self, record: QARecord) -> None:
        await self.ensure_collection()
        vector = await embed_document(record.question)
        await self._client.upsert(
            self._collection,
            points=[
                models.PointStruct(
                    id=record.id,
                    vector=vector,
                    payload={
                        "page_content": record.answer,
                        "metadata": {
                            "source_doc_id": record.id,
                            "doc_type": "qa",
                            "source": "qa",
                            "application": record.application,
                            "question": record.question,
                        },
                    },
                )
            ],
        )

    async def delete(self, point_id: str) -> None:
        await self._client.delete(
            self._collection,
            points_selector=models.PointIdsList(points=[point_id]),
        )

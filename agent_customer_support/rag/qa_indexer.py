from qdrant_client import AsyncQdrantClient, models

from agent_customer_support.applications import to_slug
from agent_customer_support.config import get_settings
from agent_customer_support.models import QARecord
from agent_customer_support.rag.embeddings import embed_document
from agent_customer_support.rag_client import _normalize_collection


# Payload keys RagClient filters on. The Qdrant deployment runs strict mode with
# `unindexed_filtering_retrieve=False`, so a filter on an unindexed key fails outright
# with a 400 — these indexes are required for scoped Q&A search to work at all, not a
# performance nicety.
_FILTERABLE_KEYS = ("metadata.application", "metadata.doc_type")


class QAIndexer:
    def __init__(self, client: AsyncQdrantClient | None = None) -> None:
        cfg = get_settings()
        self._client = client or AsyncQdrantClient(
            url=cfg.qdrant_endpoint, api_key=cfg.qdrant_api_key
        )
        self._collection = _normalize_collection(cfg.qa_collection)
        self._ready = False

    async def ensure_collection(self) -> None:
        """Create the collection and its payload indexes if absent.

        Index creation deliberately runs for pre-existing collections too, not just
        freshly created ones: the qa collection was created before these indexes were
        introduced, so gating on `create_collection` would leave it permanently
        unindexed and every scoped Q&A search failing. `create_payload_index` is
        idempotent, and `_ready` keeps it to once per process rather than once per
        upsert.
        """
        if self._ready:
            return
        if not await self._client.collection_exists(self._collection):
            await self._client.create_collection(
                self._collection,
                vectors_config=models.VectorParams(
                    size=get_settings().embedding_dim, distance=models.Distance.COSINE
                ),
            )
        for key in _FILTERABLE_KEYS:
            await self._client.create_payload_index(
                self._collection,
                field_name=key,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        self._ready = True

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
                            # Slug-normalised to match the product corpus and the
                            # read-side filter. CS types this free-text in the admin
                            # UI, so it arrives as a display name as often as a slug;
                            # normalising on write also fixes records already stored
                            # with a display name, which an admin-only fix would not.
                            # None stays None — untagged Q&A is global by design.
                            "application": to_slug(record.application)
                            if record.application
                            else None,
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

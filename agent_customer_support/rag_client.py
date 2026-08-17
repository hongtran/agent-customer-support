from collections import defaultdict
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from agent_customer_support.applications import to_slugs
from agent_customer_support.config import get_settings
from agent_customer_support.observability import tracing
from agent_customer_support.rag.embeddings import embed_query


def _meta(point: Any) -> dict:
    """Metadata dict for a Qdrant point (langchain_qdrant nests it under 'metadata')."""
    return (point.payload or {}).get("metadata", {})


def _text(point: Any) -> str:
    return (point.payload or {}).get("page_content", "")


def _normalize_collection(name: str) -> str:
    """Resolve a logical collection name to the physical Qdrant collection.

    Mirrors enterprise-llm-service's RagManager, which stores base names in
    config but indexes/queries under a versioned name
    (`name.replace("_v2", "") + "_v3"`; see
    ../enterprise-llm-service/enterprise_llm_service/rag/__init__.py). The
    legacy /rag/query applied this transform server-side, so the in-repo read
    path must apply it too or it queries a non-existent collection. Already
    `_v3`-suffixed names are passed through unchanged.
    """
    if name.endswith("_v3"):
        return name
    return name.replace("_v2", "") + "_v3"


def _build_filter(doc_type: str | None, applications: list[str] | None) -> models.Filter | None:
    """Server-side payload filter for the vector search.

    Applied by Qdrant during retrieval, not after it. Filtering in Python meant
    scoping only ever saw the top `limit` hits by similarity, so a rarely-matching
    application could be squeezed out of the candidate set entirely.

    langchain_qdrant nests chunk metadata under a `metadata` key, hence the
    `metadata.<field>` paths.

    `applications` must already be slug-normalised (see `search`).

    Applications are a hard scope, with one deliberate exception: a document with
    no `application` is global and stays visible to everyone. That covers shared
    guides and Q&A records CS approved without tagging an application
    (`QARecord.application` is optional).

    NOTE: this Qdrant runs with strict mode (`unindexed_filtering_retrieve=False`),
    so every key referenced here MUST have a keyword payload index or the query
    fails with a 400 — it does not silently fall back to a scan.
    """
    must: list[models.Condition] = []
    if doc_type:
        must.append(
            models.FieldCondition(key="metadata.doc_type", match=models.MatchValue(value=doc_type))
        )
    if applications:
        must.append(
            models.Filter(
                should=[
                    # MatchAny matches whether `application` is stored as a scalar
                    # or a list, so this is safe against either payload shape.
                    models.FieldCondition(
                        key="metadata.application",
                        match=models.MatchAny(any=applications),
                    ),
                    # is_empty covers a missing key / empty list, is_null an
                    # explicit null — we don't assume which form the indexer wrote.
                    models.IsEmptyCondition(
                        is_empty=models.PayloadField(key="metadata.application")
                    ),
                    models.IsNullCondition(is_null=models.PayloadField(key="metadata.application")),
                ]
            )
        )
    return models.Filter(must=must) if must else None


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
        per_doc: int | None = 2,
    ) -> dict:
        # `per_doc` controls how chunks are collapsed by source document, and only
        # takes effect on a GLOBAL search (no `applications` scope):
        #   - per_doc=N  -> keep at most N chunks per source_doc_id, then take top_k.
        #                   This is the product-collection default: it stops one guide
        #                   from crowding out every other document.
        #   - per_doc=None -> no collapsing; take the top_k chunks as ranked. The QA
        #                   collection passes this because it is always global and each
        #                   record is its own short document.
        # A specific-application search always skips collapsing regardless of per_doc:
        # that scope is effectively a single guide, so one chunk per document would
        # leave the agent with a single passage.
        resolved_collection = _normalize_collection(collection)
        # Callers pass display names (the widget forwards whatever the UI showed);
        # Qdrant stores slugs. Translate once here, before anything uses the value,
        # so every retrieval path is scoped on values that can actually match.
        resolved_applications = to_slugs(applications)
        with tracing.span(
            "rag.search",
            input={
                "query": query,
                "collection": resolved_collection,
                "applications": resolved_applications or [],
            },
        ) as sp:
            vec = await embed_query(query)
            resp = await self._client.query_points(
                collection_name=resolved_collection,
                query=vec,
                # Over-fetch so the per-document cap (global product search) still has
                # enough candidates to fill top_k after collapsing repeats of one
                # document. Harmless for the no-cap paths — they just slice top_k.
                # Not related to filtering — Qdrant handles that server-side.
                limit=top_k * 4,
                query_filter=_build_filter(doc_type, resolved_applications),
                with_payload=True,
            )
            points = resp.points

            above = sorted(
                ((p, p.score) for p in points if p.score >= score_threshold),
                key=lambda x: x[1],
                reverse=True,
            )

            if resolved_applications or per_doc is None:
                # Specific-application scope, or a global search that opts out of
                # collapsing (QA): keep the top_k chunks as ranked.
                ranked = above[:top_k]
            else:
                # Global product search: keep at most `per_doc` chunks per source
                # document so one guide can't crowd out the rest, then take top_k.
                counts: dict[str, int] = defaultdict(int)
                kept: list[tuple[Any, float]] = []
                for p, s in above:
                    m = _meta(p)
                    sid = m.get("doc_id", "")
                    if counts[sid] < per_doc:
                        counts[sid] += 1
                        kept.append((p, s))
                ranked = kept[:top_k]

            passages = [_text(p) for p, _ in ranked]
            metas = [{**_meta(p), "confidence": round(float(s), 4)} for p, s in ranked]
            confs = [m["confidence"] for m in metas]
            citations = sorted(
                {
                    m.get("doc_id", "")
                    for m in metas
                    if m.get("doc_id")
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
                # Per-hit metadata, positionally aligned with `passages`. Agents ignore
                # it — it exists so evaluation can score *which* documents came back
                # (application scope, source doc, rank) without re-implementing this
                # search and drifting from what the agent actually sees.
                "metas": metas,
            }
            sp.update(
                output={
                    "top_confidence": top_conf,
                    "n_passages": len(passages),
                    "citations": citations,
                }
            )
            return result


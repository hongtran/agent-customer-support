# Bring `/rag/query` in-house (Qdrant read path)

**Date:** 2026-06-21
**Status:** Approved design — ready for implementation plan
**Spec:** #1 of 2 (foundation). Spec #2 = the Q&A learning loop, which depends on this.

## Goal

Replace the external `POST {RAG_BASE_URL}/rag/query` HTTP call with a direct,
in-repo Qdrant read path that returns **results equivalent to the original**.
The agent layer (`KnowledgeAgent`, `agent/tools.py`) must keep working with no
behavioral change.

## Non-goals

- Ingestion / indexing — stays in `enterprise-llm-service`. The `cenlab`
  product collection is indexed there (one-time + ongoing) and this repo only
  reads it.
- The Q&A learning loop (capture sources, CS management UI, approval,
  second-class retrieval) — that is Spec #2 and builds on this.
- Re-hosting or copying the vector data. We connect to the **existing,
  populated Qdrant** that `enterprise-llm-service` already writes to.

## Context: what the original does

`enterprise-llm-service` owns both index and read today.

- **Embedding model:** Google `models/gemini-embedding-001`, COSINE distance
  (`enterprise_llm_service/rag/indexing.py`).
- **Vector store:** Qdrant via `QdrantClient(url=QDRANT_ENDPOINT,
  api_key=QDRANT_API_KEY)`; documents added through
  `langchain_qdrant.QdrantVectorStore`.
- **Read logic:** `precision_retrieval` (`enterprise_llm_service/rag/querying.py`):
  1. fetch `top_k * 4` candidates via `similarity_search_with_score`
  2. keep `score >= 0.4` (`DEFAULT_SCORE_THRESHOLD`); if none clear it, relax to
     floor `0.25` (`MIN_SCORE_THRESHOLD`)
  3. filter `doc_type` AND `application` (metadata fields); silently relax the
     filter if it removes everything
  4. dedup: keep highest-scoring chunk per `source_doc_id` (fallback `doc_id`)
  5. rank by score desc, cap at `top_k`; `confidence` = `round(score, 4)`

This repo's existing `RagClient` already consumes that output shape, so the
response contract is fixed and known.

## Parity decisions (resolved)

- **Score semantics:** langchain `similarity_search_with_score` returns the
  **raw Qdrant cosine score**, identical to native `qdrant-client`
  `query_points`. The `0.4` / `0.25` thresholds therefore carry over unchanged.
- **Embedding SDK:** use the **already-present `google-genai`** SDK (this repo
  has `google-genai = "^2.8.0"`; no LangChain in the tree). Parity is guaranteed
  by pinning the two hidden knobs:
  - `task_type="RETRIEVAL_QUERY"` (queries; documents were indexed as
    `RETRIEVAL_DOCUMENT`)
  - `output_dimensionality` = the live collection's vector size (authoritative —
    read from Qdrant, do not hardcode blindly)
- **Payload layout:** `langchain_qdrant` stores chunk text under payload key
  `page_content` and metadata nested under `metadata`. Native reads must use
  `payload["page_content"]` and
  `payload["metadata"][...]` for `source_doc_id` / `doc_id` / `doc_type` /
  `application`. Verified against a live point before finalizing.

## Components

### 1. `agent_customer_support/rag/embeddings.py` (new)

- `embed_query(text: str) -> list[float]` using `google-genai`, model
  `gemini-embedding-001`, with `task_type="RETRIEVAL_QUERY"` and
  `output_dimensionality` from config.
- Reads `GOOGLE_API_KEY`.
- One clear purpose: text → query vector. No Qdrant knowledge.

### 2. `RagClient.search()` rewrite (native `qdrant-client`)

- **Signature and return dict unchanged** — drop-in for `ctx.rag`.
- Flow: `embed_query(query)` → `client.query_points(collection, query=vec,
  limit=top_k*4, with_payload=True)` → replicate the 5-step
  `precision_retrieval` exactly → map to existing dict (`passages`,
  `citations`, `top_confidence`, `grounding_note`).
- Filtering may be done client-side (mirroring the original's post-fetch filter +
  silent relax) to match behavior exactly, rather than a Qdrant-native filter
  that would change the candidate set.
- Depends on: embeddings module, `qdrant-client`, config.

### 3. `config.py`

- Add: `qdrant_endpoint`, `qdrant_api_key`, `google_api_key`, `embedding_model`
  (default `gemini-embedding-001`), `embedding_dim`, `rag_score_threshold`
  (0.4), `rag_score_floor` (0.25).
- Retire `rag_base_url` (and the `RAG_BASE_URL` env) once the HTTP path is gone.

### 4. Dependencies & infra

- Add `qdrant-client` to `pyproject.toml`.
- No new `docker-compose` service: local dev points `QDRANT_ENDPOINT` at the
  **existing populated Qdrant** used by `enterprise-llm-service`.
- `.env-example`: add `GOOGLE_API_KEY`; `QDRANT_ENDPOINT` / `QDRANT_API_KEY`
  already present; remove `RAG_BASE_URL`.

### 5. Testing

- **`RagClient` unit tests:** `QdrantClient(location=":memory:")` seeded with a
  few points in the langchain payload layout (`page_content` + nested
  `metadata`), `embed_query` monkeypatched to return fixed vectors.
  Assert: threshold cutoff, floor relax, `doc_type`/`application` filter +
  silent relax, dedup-by-`source_doc_id`, ranking, `top_k` cap, and the mapped
  response dict.
- Rewrite `tests/test_rag_client.py` (currently mocks `httpx`) onto this path.
- **Agent-layer tests unchanged** — they already mock `ctx.rag`.

## Parity verification (explicit, not assumed)

Before declaring done, a smoke check (script or test) runs the **same query**
through the old external `/rag/query` and the new in-repo path against the
**same collection**, and diffs:

- the returned `source_doc_id` set, and
- the per-doc `confidence` scores (within a small float tolerance).

This proves "same results" rather than assuming it.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Embedding params drift (task_type / dims) | Read vector size from live collection; pin task_type; parity smoke test |
| Payload layout assumption wrong | Inspect a live point before finalizing the mapping |
| Filter behavior differs from original | Replicate client-side post-fetch filter + silent relax, not a Qdrant-native filter |
| Score semantics differ | Resolved — both paths use raw cosine |

## Rollout

1. Land embeddings module + `RagClient` rewrite behind unchanged signature.
2. Run parity smoke test against the live collection.
3. Remove the external HTTP path and `RAG_BASE_URL`.
4. Agent behavior unchanged throughout (contract preserved).

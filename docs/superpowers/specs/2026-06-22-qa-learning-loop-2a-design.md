# Q&A Learning Loop — Spec 2a: Capture, Curation & Indexing

**Date:** 2026-06-22
**Status:** Approved design — ready for implementation plan
**Spec:** 2a of 2 (Spec 2b = retrieval merge, depends on this).
**Builds on:** Spec 1 (in-repo Qdrant read path) — `2026-06-21-rag-search-in-repo-design.md`.

## Goal

Let CS turn questions the agent couldn't answer (or answered wrong) into a
curated, searchable Q&A corpus. CS reviews captured questions, writes/edits the
correct answer, and approves; approved Q&A is embedded and indexed into a Qdrant
`qa` collection. Retrieval/merge into the agent's answers is **Spec 2b**.

## Non-goals (→ Spec 2b)

- Querying the `qa` collection at answer time, merging with product results,
  source tagging, and guide-wins-on-conflict precedence.
- Automatic/semantic dedup of similar questions (CS handles dedup manually in
  2a; a similarity-suggestion helper is a noted future enhancement).
- Per-approver user accounts (2a uses a single shared admin token).

## Data flow

```
capture (A auto miss / B 👎 feedback / C CS manual)
   → QARecord(status="pending")            [DynamoDB acs_qa]
   → CS reviews + writes answer + approve  [/admin UI → /admin/qa API]
   → embed(question) + upsert              [Qdrant qa collection]
rejected → never indexed ; archived → point deleted
```

## Components

### 1. `QARecord` model (`models.py`)

```python
class QARecord(BaseModel):
    id: str
    question: str                      # CS-editable canonical question
    answer: str = ""                   # CS-provided; empty until filled
    status: Literal["pending", "approved", "rejected", "archived"] = "pending"
    source: Literal["cannot_answer", "feedback", "manual"]
    application: str | None = None     # optional tag (app-scoped retrieval in 2b)
    customer_id: str | None = None
    conversation_id: str | None = None
    feedback_message_id: str | None = None  # assistant turn id (feedback source); dedups repeat 👎
    bad_answer: str | None = None      # agent's wrong answer (feedback source)
    transcript: str = ""               # context for CS
    qdrant_point_id: str | None = None # = id once indexed; enables re-index/delete
    indexed_at: datetime | None = None
    approved_by: str | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
```

**Lifecycle.** `pending` → (CS edits `answer`) → `approved` (indexed). `pending`
→ `rejected` (never indexed). `approved` → edit → re-embed + re-upsert (same
point id). `approved` → `archived` (point deleted). Manual entries (C) may be
approved immediately once they have an answer.

### 2. `QAStore` (DynamoDB, table `acs_qa`)

Follows the existing one-class-per-table pattern (cf. `RequestBacklog`). Methods:
`init()` (ensure table, key `id`), `add(record)`, `get(id)`,
`list(status: str | None)`, `update(record)`, `delete(id)`. Listing uses a
`scan` with an optional `status` filter — volume is low (curated FAQs). A
`status` GSI is a future optimization, **not built** here.

Config: add `table_qa: str = "acs_qa"` to `Settings`.

### 3. Capture paths

- **A — `cannot_answer`.** In `KnowledgeAgent`, on a confirmed miss (where it
  already logs `RequestBacklog` `how_to_missing`), additionally create
  `QARecord(source="cannot_answer", status="pending", question=ctx.message,
  customer_id, conversation_id, transcript=ctx.transcript)`. The backlog entry
  stays (ops log); the QA record is the curation queue. The `KnowledgeAgent`
  gains a `QAStore` dependency via `TurnContext` (alongside `backlog`).

- **B — `feedback` (👎).** New `POST /widget/feedback`:
  ```json
  { "conversation_id": "...", "message_id": "...", "signal": "down" }
  ```
  Loads the conversation, locates the assistant turn by `message_id` and its
  immediately preceding user turn, and creates
  `QARecord(source="feedback", status="pending", question=<user turn>,
  bad_answer=<assistant turn>, customer_id, conversation_id, transcript)`.
  Returns `{ "ok": true }` (idempotent-friendly; a repeat 👎 on the same
  message updates rather than duplicates — matched by `conversation_id` +
  `message_id` stored on the record via an added `feedback_message_id` field).

  **Message id plumbing.** Add `id: str` to `Turn` (default a uuid) and
  `message_id: str` to `ChatResponse` (= the assistant turn's id). The widget
  renders a 👎 control on assistant messages that posts `message_id` back.

- **C — `manual`.** `POST /admin/qa` creates `QARecord(source="manual")` with
  CS-supplied `question`/`answer`/`application`; CS can approve immediately.

### 4. Indexing write path (new)

- **`rag/embeddings.py`** gains `embed_document(text) -> list[float]` — sibling
  to `embed_query`, identical except `task_type="RETRIEVAL_DOCUMENT"` and the
  same `output_dimensionality=embedding_dim`. (Refactor the shared call into a
  private `_embed(text, task_type)` used by both.)

- **`rag/qa_indexer.py`** (new) — `QAIndexer`:
  - `__init__(client: AsyncQdrantClient | None = None)` — mirrors `RagClient`.
  - `async ensure_collection()` — create the qa collection (size
    `embedding_dim`, `Distance.COSINE`) if absent.
  - `async upsert(record: QARecord)` — vector = `embed_document(record.question)`
    (**question only**); point id = `record.id`; payload:
    ```python
    {
        "page_content": record.answer,          # answer is the returned passage
        "metadata": {
            "source_doc_id": record.id,
            "doc_type": "qa",
            "source": "qa",
            "application": record.application,
            "question": record.question,        # kept for reference
        },
    }
    ```
  - `async delete(point_id: str)` — remove the point.
  - Collection name resolves through the existing
    `rag_client._normalize_collection` (`_v3` convention) and uses the **same
    payload layout** as the product collection, so Spec 2b reads it with
    `RagClient.search` and **zero new retrieval code**.

  Config: add `qa_collection: str` (base name) to `Settings`.

  **Rationale for layout (a):** vector is question-only (best question-to-question
  matching); the answer lives in `page_content` because `RagClient.search`
  returns `page_content` as the passage — putting the answer there is what lets
  2b reuse the Spec-1 read path unchanged.

### 5. CS management API (`channels/admin.py`, new router)

`APIRouter(prefix="/admin", tags=["admin"])`, mounted in `server.py`. Every
route depends on an auth guard that checks `X-Admin-Token == settings.admin_token`
(raise 401 otherwise). Config: add `admin_token: str` to `Settings`
(empty default ⇒ guard rejects all, so it must be set to enable admin).

| Method & path | Action |
|---|---|
| `GET /admin/qa?status=pending` | list records (optional status filter) |
| `GET /admin/qa/{id}` | fetch one |
| `POST /admin/qa` | create (manual, `source="manual"`) |
| `PATCH /admin/qa/{id}` | edit `question`/`answer`/`application`; if record is `approved`, re-upsert |
| `POST /admin/qa/{id}/approve` | requires non-empty `answer`; set `approved`, `approved_by`, `indexed_at`; `QAIndexer.upsert` |
| `POST /admin/qa/{id}/reject` | set `rejected` (no index) |
| `POST /admin/qa/{id}/archive` | set `archived`; `QAIndexer.delete` |

### 6. CS admin UI (`ui/app/admin/page.tsx`)

Minimal but usable, reusing existing styling/components:
- Admin token entered once, stored in `localStorage`, sent as `X-Admin-Token`.
- Pending-queue table (question, source, created_at) → select a record →
  detail panel showing question, `bad_answer` (if any), transcript, and an
  editable answer field + optional application → **Approve / Reject** buttons.
- A "New Q&A" form for manual (C) entries.
- API access via `ui/lib/api.ts` (extend with admin calls).

### 7. Error handling

- Admin routes: 401 on bad/missing token; 404 on unknown id; 409 on approve
  with empty answer.
- Indexing on approve is index-first: call `QAIndexer.upsert` and only persist
  `status="approved"` to the store if it succeeds. On a Qdrant/embedding failure,
  return 502 and leave the record `pending` (never half-approved).
- Feedback endpoint: 404 if `conversation_id`/`message_id` not found; never
  errors the chat path.

## Testing

- **`QAStore`** — DynamoDB Local: add/get/update/delete; `list(status=...)`
  filtering.
- **`embed_document`** — param test asserting `task_type="RETRIEVAL_DOCUMENT"`
  and `output_dimensionality=embedding_dim` (mirrors the `embed_query` test).
- **`QAIndexer`** — in-memory Qdrant (`location=":memory:"`): `ensure_collection`
  creates it; `upsert` writes a point whose id = `record.id`, payload
  `page_content == answer`, `metadata.doc_type == "qa"`, and the vector came from
  the **question** (monkeypatch `embed_document` to a sentinel and assert it was
  called with `record.question`); `delete` removes the point.
- **Admin API** — FastAPI `TestClient`: token guard (401 without header);
  create/list/edit; approve calls indexer + flips status; approve with empty
  answer → 409; archive deletes the point.
- **Feedback endpoint** — posting `signal:"down"` for a known `message_id`
  creates a `feedback` `QARecord` with the right `question`/`bad_answer`; repeat
  post updates rather than duplicates; unknown id → 404.
- **`KnowledgeAgent`** — a confirmed miss creates a pending `cannot_answer`
  `QARecord` (in addition to the existing backlog entry).

## Out of scope (→ Spec 2b)

Retrieval merge in `KnowledgeAgent`: query the `qa` collection alongside the
product collection, tag results by `source`, and let the product guide win on
conflict. The `qa` collection's layout is designed so 2b needs no new retrieval
code (reuses `RagClient.search`).

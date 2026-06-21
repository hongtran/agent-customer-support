# RAG Search In-Repo (Qdrant Read Path) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the external `POST {RAG_BASE_URL}/rag/query` HTTP call with a direct in-repo Qdrant read path that returns results equivalent to the original, with no behavioral change to the agent layer.

**Architecture:** A new `rag/embeddings.py` embeds the query with Google `gemini-embedding-001` (via the already-present `google-genai` SDK, async). `RagClient.search()` is rewritten to query the existing populated Qdrant directly with `AsyncQdrantClient`, replicating the original `precision_retrieval` logic (threshold → floor relax → filter → dedup → rank), and returns the **same** response dict. The method signature is unchanged, so `KnowledgeAgent`, `agent/tools.py`, and agent-layer tests are untouched.

**Tech Stack:** Python 3.13, `google-genai` (already a dep), `qdrant-client` (new), pytest + pytest-asyncio.

## Global Constraints

- Connect to the **existing populated Qdrant** used by `enterprise-llm-service`; this repo does **not** ingest/index.
- Embedding model: `gemini-embedding-001`, `task_type="RETRIEVAL_QUERY"`, `output_dimensionality` = the live collection's vector size (from config, default verified against live collection).
- Distance metric: COSINE. Scores are raw Qdrant cosine values.
- Effective default `score_threshold = 0.6` (preserve current `RagClient` default — it overrode the service's 0.4). Internal floor `MIN_SCORE_THRESHOLD = 0.25`.
- Payload layout (langchain_qdrant): chunk text under payload key `page_content`; metadata nested under payload key `metadata`.
- `RagClient.search()` signature, return-dict keys (`passages`, `citations`, `top_confidence`, `grounding_note`), tracing span name (`rag.search`), and the `grounding_note` Vietnamese string are preserved verbatim.

---

### Task 1: Config + dependency

**Files:**
- Modify: `pyproject.toml` (add `qdrant-client`)
- Modify: `agent_customer_support/config.py`
- Modify: `.env-example`
- Test: `tests/test_config_rag.py` (create)

**Interfaces:**
- Produces: `Settings.qdrant_endpoint: str`, `Settings.qdrant_api_key: str`, `Settings.google_api_key: str`, `Settings.embedding_model: str`, `Settings.embedding_dim: int`. (Consumed by Tasks 2 and 3.)

- [ ] **Step 1: Add the dependency**

Run: `poetry add qdrant-client`
Expected: `qdrant-client` added under `[tool.poetry.dependencies]`, lockfile updated.

- [ ] **Step 2: Write the failing test**

Create `tests/test_config_rag.py`:

```python
from agent_customer_support.config import Settings


def test_rag_settings_have_expected_defaults():
    s = Settings()
    assert s.qdrant_endpoint  # from QDRANT_ENDPOINT env stub
    assert s.embedding_model == "gemini-embedding-001"
    assert s.embedding_dim == 3072
    assert hasattr(s, "google_api_key")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `poetry run pytest tests/test_config_rag.py -v`
Expected: FAIL with `AttributeError` (fields not defined).

- [ ] **Step 4: Add settings fields**

In `agent_customer_support/config.py`, inside `class Settings`, replace the line `rag_base_url: str = "http://localhost:7799"` with the Qdrant/embedding block (remove `rag_base_url`):

```python
    # RAG (Qdrant read path)
    qdrant_endpoint: str = "http://localhost:6333"
    qdrant_api_key: str = "dummy"
    google_api_key: str = ""
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 3072
    product_collection: str = "cenlab"
```

(Keep the existing `product_collection` line only once — if it already exists below, do not duplicate it; move it into this block.)

- [ ] **Step 5: Update `.env-example`**

In `.env-example`, under `# RAG`, remove the `RAG_BASE_URL=...` line and ensure these exist:

```
# RAG (Qdrant read path)
QDRANT_ENDPOINT=http://localhost:6333
QDRANT_API_KEY=dummy
GOOGLE_API_KEY=
```

- [ ] **Step 6: Run test to verify it passes**

Run: `poetry run pytest tests/test_config_rag.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml poetry.lock agent_customer_support/config.py .env-example tests/test_config_rag.py
git commit -m "feat(rag): add Qdrant + embedding settings, drop rag_base_url

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Embeddings module

**Files:**
- Create: `agent_customer_support/rag/__init__.py`
- Create: `agent_customer_support/rag/embeddings.py`
- Test: `tests/rag/test_embeddings.py` (create), `tests/rag/__init__.py` (create, empty)

**Interfaces:**
- Consumes: `Settings.google_api_key`, `Settings.embedding_model`, `Settings.embedding_dim` (Task 1).
- Produces: `async def embed_query(text: str) -> list[float]`. (Consumed by Task 3.)

- [ ] **Step 1: Write the failing test**

Create `tests/rag/__init__.py` (empty) and `tests/rag/test_embeddings.py`:

```python
import pytest
import agent_customer_support.rag.embeddings as emb

pytestmark = pytest.mark.asyncio


async def test_embed_query_calls_gemini_with_retrieval_query(monkeypatch):
    captured = {}

    class FakeEmbeddings:
        def __init__(self, values):
            self.embeddings = [type("E", (), {"values": values})()]

    class FakeAio:
        class models:
            @staticmethod
            async def embed_content(*, model, contents, config):
                captured["model"] = model
                captured["contents"] = contents
                captured["task_type"] = config.task_type
                captured["dim"] = config.output_dimensionality
                return FakeEmbeddings([0.1, 0.2, 0.3])

    class FakeClient:
        aio = FakeAio()

    monkeypatch.setattr(emb, "_client", lambda: FakeClient())

    vec = await emb.embed_query("cách tạo mẫu")

    assert vec == [0.1, 0.2, 0.3]
    assert captured["model"] == "gemini-embedding-001"
    assert captured["contents"] == "cách tạo mẫu"
    assert captured["task_type"] == "RETRIEVAL_QUERY"
    assert captured["dim"] == 3072
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/rag/test_embeddings.py -v`
Expected: FAIL with `ModuleNotFoundError: agent_customer_support.rag.embeddings`.

- [ ] **Step 3: Write the implementation**

Create `agent_customer_support/rag/__init__.py` (empty).

Create `agent_customer_support/rag/embeddings.py`:

```python
from functools import lru_cache

from google import genai
from google.genai import types

from agent_customer_support.config import get_settings


@lru_cache
def _client() -> genai.Client:
    return genai.Client(api_key=get_settings().google_api_key)


async def embed_query(text: str) -> list[float]:
    """Embed a search query with the same model/params used to index the
    collection, so the query vector lands in the same space.

    task_type RETRIEVAL_QUERY mirrors how documents were indexed
    (RETRIEVAL_DOCUMENT); output_dimensionality must match the collection's
    vector size or the search mismatches.
    """
    cfg = get_settings()
    resp = await _client().aio.models.embed_content(
        model=cfg.embedding_model,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=cfg.embedding_dim,
        ),
    )
    return list(resp.embeddings[0].values)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/rag/test_embeddings.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent_customer_support/rag/__init__.py agent_customer_support/rag/embeddings.py tests/rag/__init__.py tests/rag/test_embeddings.py
git commit -m "feat(rag): query embedding via google-genai (gemini-embedding-001)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Rewrite `RagClient.search()` onto native Qdrant

**Files:**
- Modify: `agent_customer_support/rag_client.py` (full rewrite)
- Test: `tests/test_rag_client.py` (full rewrite)

**Interfaces:**
- Consumes: `embed_query` (Task 2); `Settings.qdrant_endpoint`, `Settings.qdrant_api_key` (Task 1).
- Produces: `RagClient(client: AsyncQdrantClient | None = None)` and unchanged `async def search(self, query, collection, top_k=8, score_threshold=0.6, doc_type=None, applications=None) -> dict` returning `{"passages", "citations", "top_confidence", "grounding_note"}`.

- [ ] **Step 1: Write the failing tests**

Replace the entire contents of `tests/test_rag_client.py` with:

```python
import pytest
from contextlib import contextmanager
from unittest.mock import MagicMock

from qdrant_client import AsyncQdrantClient, models

from agent_customer_support.rag_client import RagClient
import agent_customer_support.rag_client as rag_client_mod

pytestmark = pytest.mark.asyncio

COLLECTION = "cenlab"


def _point(pid, vector, *, source_doc_id, doc_type="guide", application="Lab", text="noi dung"):
    return models.PointStruct(
        id=pid,
        vector=vector,
        payload={
            "page_content": text,
            "metadata": {
                "source_doc_id": source_doc_id,
                "doc_type": doc_type,
                "application": application,
            },
        },
    )


async def _client_with(points):
    client = AsyncQdrantClient(location=":memory:")
    await client.create_collection(
        COLLECTION,
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    await client.upsert(COLLECTION, points=points)
    return client


@pytest.fixture(autouse=True)
def _stub_embed(monkeypatch):
    async def fake_embed(text):
        return [1.0, 0.0]

    monkeypatch.setattr(rag_client_mod, "embed_query", fake_embed)


async def test_threshold_keeps_only_scores_at_or_above_default():
    # q=[1,0]; A->cos 1.0, C->cos 0.6, D->cos 0.3 (below 0.6 default)
    client = await _client_with([
        _point(1, [1.0, 0.0], source_doc_id="hdsd#1", text="Buoc 1"),
        _point(2, [0.6, 0.8], source_doc_id="hdsd#2", text="Buoc 2"),
        _point(3, [0.3, 0.9539392], source_doc_id="hdsd#3", text="Buoc 3"),
    ])
    res = await RagClient(client=client).search("q", collection=COLLECTION)
    assert res["top_confidence"] == 1.0
    assert set(res["citations"]) == {"hdsd#1", "hdsd#2"}  # hdsd#3 below 0.6


async def test_floor_relax_when_nothing_clears_threshold():
    client = await _client_with([
        _point(1, [0.6, 0.8], source_doc_id="hdsd#2"),
        _point(2, [0.3, 0.9539392], source_doc_id="hdsd#3"),
    ])
    # threshold 0.9 clears nothing -> relax to 0.25 floor -> both (>=0.25) returned
    res = await RagClient(client=client).search("q", collection=COLLECTION, score_threshold=0.9)
    assert set(res["citations"]) == {"hdsd#2", "hdsd#3"}


async def test_application_filter_then_silent_relax():
    client = await _client_with([
        _point(1, [1.0, 0.0], source_doc_id="hdsd#1", application="Lab"),
        _point(2, [0.6, 0.8], source_doc_id="hdsd#2", application="Other"),
    ])
    res = await RagClient(client=client).search(
        "q", collection=COLLECTION, applications=["Lab"]
    )
    assert set(res["citations"]) == {"hdsd#1"}

    # filter that matches nothing is silently relaxed to unfiltered
    res2 = await RagClient(client=client).search(
        "q", collection=COLLECTION, applications=["DoesNotExist"]
    )
    assert set(res2["citations"]) == {"hdsd#1", "hdsd#2"}


async def test_dedup_keeps_highest_scoring_chunk_per_source():
    client = await _client_with([
        _point(1, [1.0, 0.0], source_doc_id="hdsd#1", text="best"),
        _point(2, [0.6, 0.8], source_doc_id="hdsd#1", text="worse"),
    ])
    res = await RagClient(client=client).search("q", collection=COLLECTION)
    assert res["citations"] == ["hdsd#1"]
    assert res["passages"] == ["best"]
    assert res["top_confidence"] == 1.0


async def test_grounding_note_is_neutral_hint():
    client = await _client_with([
        _point(1, [1.0, 0.0], source_doc_id="hdsd#1"),
    ])
    res = await RagClient(client=client).search("q", collection=COLLECTION)
    note = res["grounding_note"]
    assert "log_request" not in note
    assert "clarification" not in note
    assert "confidence" in note.lower()


async def test_search_invokes_tracing_span(monkeypatch):
    handle = MagicMock()
    calls: dict = {}

    @contextmanager
    def fake_span(name, *, input=None, metadata=None):
        calls["name"] = name
        yield handle

    monkeypatch.setattr(rag_client_mod.tracing, "span", fake_span)

    client = await _client_with([
        _point(1, [1.0, 0.0], source_doc_id="hdsd#5"),
    ])
    res = await RagClient(client=client).search("q", collection=COLLECTION)

    assert calls["name"] == "rag.search"
    handle.update.assert_called_once()
    assert res["top_confidence"] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_rag_client.py -v`
Expected: FAIL (current `RagClient` takes `base_url`, uses httpx; no `embed_query` attr on module).

- [ ] **Step 3: Write the implementation**

Replace the entire contents of `agent_customer_support/rag_client.py` with:

```python
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
                output={"top_confidence": top_conf, "n_passages": len(passages), "citations": citations}
            )
            return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_rag_client.py -v`
Expected: PASS (all 6 tests).

- [ ] **Step 5: Commit**

```bash
git add agent_customer_support/rag_client.py tests/test_rag_client.py
git commit -m "feat(rag): query Qdrant directly, drop external /rag/query HTTP call

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Wire-up cleanup + full suite green

**Files:**
- Modify: `tests/conftest.py`
- Verify (no change expected): `agent_customer_support/agents/coordinator.py:36` (`self.rag = RagClient()`), `agent_customer_support/agents/knowledge.py:153`

**Interfaces:**
- Consumes: everything from Tasks 1–3. Produces: green test suite with the external path fully removed.

- [ ] **Step 1: Update conftest env stubs**

In `tests/conftest.py`, remove the `RAG_BASE_URL` line and add a `GOOGLE_API_KEY` stub. The relevant block should read:

```python
os.environ.setdefault("PRODUCT_COLLECTION", "cenlab")
os.environ.setdefault("AGENT_MODEL", "gpt-4o-mini")
os.environ.setdefault("DYNAMODB_ENDPOINT_URL", "http://localhost:8000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "local")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "local")
os.environ.setdefault("AWS_REGION", "ap-southeast-1")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("QDRANT_ENDPOINT", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "local")
os.environ.setdefault("GOOGLE_API_KEY", "fake-google-for-tests")
os.environ.setdefault("OPENAI_API_KEY", "sk-fake-for-tests")
```

(Leave the Celery / TogetherAI stubs as they are.)

- [ ] **Step 2: Confirm no other references to the removed external path**

Run: `grep -rn "rag_base_url\|RAG_BASE_URL\|/rag/query\|base_url=" agent_customer_support/ tests/`
Expected: no matches (other than unrelated `base_url=` in other clients, if any — confirm none refer to RagClient).

- [ ] **Step 3: Run the full suite**

Run: `make test`
Expected: PASS. Agent-layer tests (`tests/agents/test_knowledge.py` etc.) pass unchanged because they mock `ctx.rag`. If `tests/agents/test_knowledge.py:284` asserts `ctx.rag.search.assert_awaited_once_with(standalone, collection=ANY, applications=None)`, it still holds — the signature is preserved.

- [ ] **Step 4: Lint**

Run: `make lint`
Expected: clean (ruff format + check + mypy). Fix any typing issues in the new files if reported.

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py
git commit -m "chore(rag): drop RAG_BASE_URL stub, add GOOGLE_API_KEY test stub

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Parity smoke script (manual verification against live Qdrant)

**Files:**
- Create: `scripts/rag_parity_check.py`

**Interfaces:**
- Consumes: `RagClient` (Task 3), live Qdrant + Gemini via env. Produces: a runnable script that diffs old vs new results. Not part of `make test` (needs live services).

- [ ] **Step 1: Write the script**

Create `scripts/rag_parity_check.py`:

```python
"""Parity check: compare the in-repo Qdrant read path against the legacy
external /rag/query for the same queries and collection.

Requires live env: QDRANT_ENDPOINT, QDRANT_API_KEY, GOOGLE_API_KEY, and
LEGACY_RAG_BASE_URL pointing at the still-running enterprise-llm-service.

Usage:
    poetry run python scripts/rag_parity_check.py "cách tạo mẫu xét nghiệm"
"""

import asyncio
import os
import sys

import httpx

from agent_customer_support.config import get_settings
from agent_customer_support.rag_client import RagClient

QUERIES = [
    "cách tạo mẫu xét nghiệm",
    "đổi mật khẩu",
    "khôi phục tài khoản",
]


async def legacy(query: str, collection: str) -> dict:
    base = os.environ["LEGACY_RAG_BASE_URL"]
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{base}/rag/query",
            json={
                "query": query,
                "collection_name": collection,
                "top_k": 8,
                "score_threshold": 0.6,
                "doc_type": None,
                "applications": [],
            },
        )
        resp.raise_for_status()
        data = resp.json()
    metas = data.get("metadatas", []) or []
    cites = sorted(
        {m.get("source_doc_id") or m.get("doc_id", "") for m in metas if m.get("source_doc_id") or m.get("doc_id")}
    )
    confs = {m.get("source_doc_id") or m.get("doc_id", ""): m.get("confidence", 0.0) for m in metas}
    return {"citations": cites, "confs": confs}


async def main() -> None:
    queries = sys.argv[1:] or QUERIES
    collection = get_settings().product_collection
    rag = RagClient()
    mismatches = 0
    for q in queries:
        new = await rag.search(q, collection=collection)
        old = await legacy(q, collection)
        new_cites = set(new["citations"])
        old_cites = set(old["citations"])
        same = new_cites == old_cites
        mismatches += 0 if same else 1
        print(f"\nQ: {q}")
        print(f"  new citations: {sorted(new_cites)}")
        print(f"  old citations: {sorted(old_cites)}")
        print(f"  match: {same}")
        if not same:
            print(f"  only new: {new_cites - old_cites}")
            print(f"  only old: {old_cites - new_cites}")
    print(f"\n{len(queries) - mismatches}/{len(queries)} queries matched.")
    if mismatches:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Run it against live services (manual)**

Run: `LEGACY_RAG_BASE_URL=http://localhost:7799 poetry run python scripts/rag_parity_check.py`
Expected: each query reports `match: True`; final line `N/N queries matched.` Investigate any mismatch (most likely `embedding_dim` or `task_type` drift — confirm the live collection's vector size via the Qdrant dashboard/API and set `EMBEDDING_DIM` to match).

- [ ] **Step 3: Commit**

```bash
git add scripts/rag_parity_check.py
git commit -m "test(rag): parity smoke script comparing in-repo vs legacy /rag/query

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Embeddings module (`rag/embeddings.py`, google-genai, RETRIEVAL_QUERY, output_dim) → Task 2. ✓
- `RagClient.search()` native Qdrant rewrite, same contract, 5-step precision_retrieval → Task 3. ✓
- Config (`qdrant_endpoint`, `qdrant_api_key`, `google_api_key`, `embedding_model`, retire `rag_base_url`) → Task 1. ✓
- Deps (`qdrant-client`), `.env-example` (`GOOGLE_API_KEY`) → Task 1. ✓
- Testing (`:memory:` Qdrant + monkeypatched embed; rewrite `test_rag_client.py`; agent tests unchanged) → Tasks 3, 4. ✓
- Parity verification (diff source_doc_id + scores, old vs new) → Task 5. ✓
- Payload layout (`page_content` + nested `metadata`) → `_meta`/`_text` in Task 3 + seeded test points. ✓
- Score semantics (raw cosine, 0.6 default / 0.25 floor) → Global Constraints + Task 3. ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases"; all code steps contain full code. ✓

**Type consistency:** `embed_query(text:str)->list[float]` defined in Task 2, consumed by Task 3 via module attr `rag_client_mod.embed_query` (monkeypatch target matches the `from ... import embed_query` binding). `RagClient(client=...)` constructor used in Task 3 tests matches the implementation. `search(...)` signature matches the preserved agent-layer call (`collection=`, `applications=`). ✓

**Note for executor:** `embedding_dim` default is `3072` (gemini-embedding-001 default). If the live collection was indexed at a different dimension, Task 5 will reveal it — update `EMBEDDING_DIM` env / `Settings.embedding_dim` to the live collection's vector size before declaring parity.

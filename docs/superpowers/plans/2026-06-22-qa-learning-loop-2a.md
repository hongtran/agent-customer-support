# Q&A Learning Loop 2a — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let CS capture, curate, approve, and index a Q&A corpus (the questions the agent couldn't answer or answered wrong) into a Qdrant `qa` collection.

**Architecture:** A DynamoDB-backed `QARecord` lifecycle (pending→approved/rejected/archived). Three capture sources create pending records; a token-gated `/admin/qa` API + Next.js `/admin` page let CS edit and approve; on approval a new write path embeds the question (`RETRIEVAL_DOCUMENT`) and upserts into a Qdrant `qa` collection whose payload layout matches the product collection so Spec 2b reuses `RagClient.search`.

**Tech Stack:** Python 3.13, FastAPI, DynamoDB (aioboto3), Qdrant (`qdrant-client`), `google-genai` embeddings, Next.js (TS) for the admin UI.

## Global Constraints

- Vector for a Q&A point = `embed_document(record.question)` — **question only**. The answer is NOT embedded.
- Qdrant point payload: `{"page_content": record.answer, "metadata": {"source_doc_id": id, "doc_type": "qa", "source": "qa", "application": ..., "question": ...}}`. Point id = `record.id`.
- `QARecord.id` = `str(uuid4())` (canonical UUID — Qdrant-compatible point id).
- qa collection name resolves via the existing `agent_customer_support.rag_client._normalize_collection` (`_v3` convention); config holds the base name `qa_collection`.
- Embedding: model `gemini-embedding-001`, `output_dimensionality = embedding_dim` (3072), `task_type="RETRIEVAL_DOCUMENT"` for documents.
- Approve is **index-first**: call `QAIndexer.upsert` and only persist `status="approved"` if it succeeds (502 + leave pending on failure).
- Admin routes require `X-Admin-Token == settings.admin_token`; empty `admin_token` ⇒ all admin requests rejected (401).
- Capture A keeps the existing `RequestBacklog` `how_to_missing` entry AND adds the QA record.

---

### Task 1: Models + config

**Files:**
- Modify: `agent_customer_support/models.py`
- Modify: `agent_customer_support/config.py`
- Modify: `tests/conftest.py`
- Test: `tests/test_models_qa.py` (create)

**Interfaces:**
- Produces: `QARecord` (pydantic model); `Turn.id: str`; `ChatResponse.message_id: str`; `Settings.table_qa`, `Settings.qa_collection`, `Settings.admin_token`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_models_qa.py`:

```python
from agent_customer_support.models import QARecord, Turn, ChatResponse
from agent_customer_support.config import Settings


def test_qarecord_defaults():
    rec = QARecord(question="làm sao đổi mật khẩu?", source="manual")
    assert rec.id  # auto uuid
    assert "-" in rec.id  # canonical uuid form (Qdrant-compatible)
    assert rec.status == "pending"
    assert rec.answer == ""
    assert rec.application is None


def test_turn_has_auto_id_and_chatresponse_message_id():
    t = Turn(role="assistant", content="hi")
    assert t.id
    resp = ChatResponse(conversation_id="c1", reply="hi", message_id=t.id)
    assert resp.message_id == t.id


def test_qa_settings_present():
    s = Settings()
    assert s.table_qa == "acs_qa"
    assert s.qa_collection  # from QA_COLLECTION env stub
    assert hasattr(s, "admin_token")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_models_qa.py -v`
Expected: FAIL (`ImportError: cannot import name 'QARecord'`).

- [ ] **Step 3: Implement the model + Turn/ChatResponse fields**

In `agent_customer_support/models.py`, add `from uuid import uuid4` to the imports at the top (keep existing imports).

Add `id` to `Turn` (the class currently starts with `role`):

```python
class Turn(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    role: Literal["user", "assistant"]
    content: str
    attachments: list[Attachment] = Field(default_factory=list)
    ts: datetime = Field(default_factory=_now)
```

Add `message_id` to `ChatResponse`:

```python
class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    citations: list[str] = Field(default_factory=list)
    escalated: bool = False
    message_id: str = ""
```

Add the `QARecord` model after the `RequestRecord` block:

```python
# ---- Q&A learning loop ----


class QARecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    question: str
    answer: str = ""
    status: Literal["pending", "approved", "rejected", "archived"] = "pending"
    source: Literal["cannot_answer", "feedback", "manual"]
    application: str | None = None
    customer_id: str | None = None
    conversation_id: str | None = None
    feedback_message_id: str | None = None
    bad_answer: str | None = None
    transcript: str = ""
    qdrant_point_id: str | None = None
    indexed_at: datetime | None = None
    approved_by: str | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
```

- [ ] **Step 4: Add config fields**

In `agent_customer_support/config.py`, add `table_qa` to the table-names block and `qa_collection` to the RAG block and `admin_token` near the other secrets. Concretely, in the `# RAG (Qdrant read path)` block add after `product_collection`:

```python
    qa_collection: str = "cenlab_qa"
```

In the `# table names` block add:

```python
    table_qa: str = "acs_qa"
```

And add anywhere among the top-level settings (e.g. after `zalo_cs_webhook_url`):

```python
    admin_token: str = ""
```

- [ ] **Step 5: Add test env stubs**

In `tests/conftest.py`, add after the `GOOGLE_API_KEY` line:

```python
os.environ.setdefault("QA_COLLECTION", "cenlab_qa")
os.environ.setdefault("ADMIN_TOKEN", "test-admin-token")
```

- [ ] **Step 6: Run test to verify it passes**

Run: `poetry run pytest tests/test_models_qa.py -v`
Expected: PASS (3 tests).

- [ ] **Step 7: Commit**

```bash
git add agent_customer_support/models.py agent_customer_support/config.py tests/conftest.py tests/test_models_qa.py
git commit -m "feat(qa): QARecord model, Turn.id, message_id, qa config

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `QAStore` (DynamoDB)

**Files:**
- Create: `agent_customer_support/stores/qa_store.py`
- Test: `tests/stores/test_qa_store.py` (create; `tests/stores/__init__.py` if missing)

**Interfaces:**
- Consumes: `QARecord` (Task 1); `Settings.table_qa`; `ensure_table`, `get_resource` from `stores/dynamo.py`.
- Produces: `QAStore` with `async init()`, `async add(record: QARecord) -> QARecord`, `async get(id: str) -> QARecord | None`, `async list(status: str | None = None) -> list[QARecord]`, `async update(record: QARecord) -> QARecord`, `async delete(id: str) -> None`, `async find_by_feedback_message_id(mid: str) -> QARecord | None`.

- [ ] **Step 1: Write the failing test**

Create `tests/stores/__init__.py` (empty) if it does not exist, and `tests/stores/test_qa_store.py`:

```python
import pytest
from agent_customer_support.models import QARecord
from agent_customer_support.stores.qa_store import QAStore

pytestmark = pytest.mark.asyncio


async def test_add_get_update_delete():
    store = QAStore()
    await store.init()
    rec = await store.add(QARecord(question="đổi mật khẩu?", source="manual"))
    got = await store.get(rec.id)
    assert got and got.question == "đổi mật khẩu?"

    got.answer = "Vào Cài đặt > Đổi mật khẩu"
    got.status = "approved"
    await store.update(got)
    again = await store.get(rec.id)
    assert again.answer.startswith("Vào Cài đặt")
    assert again.status == "approved"

    await store.delete(rec.id)
    assert await store.get(rec.id) is None


async def test_list_filters_by_status():
    store = QAStore()
    await store.init()
    p = await store.add(QARecord(question="q-pending", source="manual"))
    a = await store.add(QARecord(question="q-approved", source="manual", status="approved"))
    pending = await store.list(status="pending")
    ids = {r.id for r in pending}
    assert p.id in ids and a.id not in ids


async def test_find_by_feedback_message_id():
    store = QAStore()
    await store.init()
    rec = await store.add(
        QARecord(question="q", source="feedback", feedback_message_id="msg-123")
    )
    found = await store.find_by_feedback_message_id("msg-123")
    assert found and found.id == rec.id
    assert await store.find_by_feedback_message_id("nope") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/stores/test_qa_store.py -v`
Expected: FAIL (`ModuleNotFoundError: ...stores.qa_store`).

- [ ] **Step 3: Implement the store**

Create `agent_customer_support/stores/qa_store.py`:

```python
from boto3.dynamodb.conditions import Attr

from agent_customer_support.config import get_settings
from agent_customer_support.models import QARecord
from agent_customer_support.stores.dynamo import ensure_table, get_resource


class QAStore:
    def __init__(self) -> None:
        self.table_name = get_settings().table_qa

    async def init(self) -> None:
        await ensure_table(self.table_name, key="id")

    async def add(self, record: QARecord) -> QARecord:
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            await table.put_item(Item=record.model_dump(mode="json"))
        return record

    async def get(self, record_id: str) -> QARecord | None:
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            res = await table.get_item(Key={"id": record_id})
        item = res.get("Item")
        return QARecord.model_validate(item) if item else None

    async def list(self, status: str | None = None) -> list[QARecord]:
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            if status:
                res = await table.scan(FilterExpression=Attr("status").eq(status))
            else:
                res = await table.scan()
        return [QARecord.model_validate(i) for i in res.get("Items", [])]

    async def update(self, record: QARecord) -> QARecord:
        from datetime import UTC, datetime

        record.updated_at = datetime.now(UTC)
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            await table.put_item(Item=record.model_dump(mode="json"))
        return record

    async def delete(self, record_id: str) -> None:
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            await table.delete_item(Key={"id": record_id})

    async def find_by_feedback_message_id(self, mid: str) -> QARecord | None:
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            res = await table.scan(FilterExpression=Attr("feedback_message_id").eq(mid))
        items = res.get("Items", [])
        return QARecord.model_validate(items[0]) if items else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/stores/test_qa_store.py -v`
Expected: PASS (3 tests). (Requires DynamoDB Local — `make infra-up` if not running.)

- [ ] **Step 5: Commit**

```bash
git add agent_customer_support/stores/qa_store.py tests/stores/test_qa_store.py tests/stores/__init__.py
git commit -m "feat(qa): QAStore DynamoDB CRUD + status/feedback lookups

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `embed_document` (write-path embedding)

**Files:**
- Modify: `agent_customer_support/rag/embeddings.py`
- Test: `tests/rag/test_embeddings.py` (extend)

**Interfaces:**
- Produces: `async def embed_document(text: str) -> list[float]` (task_type `RETRIEVAL_DOCUMENT`). Existing `embed_query` keeps its signature/behavior.

- [ ] **Step 1: Write the failing test**

Append to `tests/rag/test_embeddings.py`:

```python
async def test_embed_document_uses_retrieval_document(monkeypatch):
    captured = {}

    class FakeEmbeddings:
        def __init__(self, values):
            self.embeddings = [type("E", (), {"values": values})()]

    class FakeAio:
        class models:
            @staticmethod
            async def embed_content(*, model, contents, config):
                captured["task_type"] = config.task_type
                captured["dim"] = config.output_dimensionality
                return FakeEmbeddings([0.4, 0.5, 0.6])

    class FakeClient:
        aio = FakeAio()

    monkeypatch.setattr(emb, "_client", lambda: FakeClient())

    vec = await emb.embed_document("Câu hỏi: ...")
    assert vec == [0.4, 0.5, 0.6]
    assert captured["task_type"] == "RETRIEVAL_DOCUMENT"
    assert captured["dim"] == 3072
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/rag/test_embeddings.py::test_embed_document_uses_retrieval_document -v`
Expected: FAIL (`AttributeError: module ... has no attribute 'embed_document'`).

- [ ] **Step 3: Refactor + add `embed_document`**

Replace the body of `agent_customer_support/rag/embeddings.py` below the `_client` definition with a shared helper and two public functions:

```python
async def _embed(text: str, task_type: str) -> list[float]:
    cfg = get_settings()
    resp = await _client().aio.models.embed_content(
        model=cfg.embedding_model,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=cfg.embedding_dim,
        ),
    )
    embeddings = resp.embeddings
    if not embeddings:
        raise ValueError("Embedding response contained no embeddings")
    values = embeddings[0].values
    if values is None:
        raise ValueError("Embedding response contained no values")
    return list(values)


async def embed_query(text: str) -> list[float]:
    """Embed a search query (RETRIEVAL_QUERY) — matches the indexed documents'
    space so search lands correctly."""
    return await _embed(text, "RETRIEVAL_QUERY")


async def embed_document(text: str) -> list[float]:
    """Embed a document/stored question (RETRIEVAL_DOCUMENT) for the qa index,
    paired with RETRIEVAL_QUERY at search time."""
    return await _embed(text, "RETRIEVAL_DOCUMENT")
```

(Keep the existing imports and `_client()` at the top of the file unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/rag/test_embeddings.py -v`
Expected: PASS (both the existing query test and the new document test).

- [ ] **Step 5: Commit**

```bash
git add agent_customer_support/rag/embeddings.py tests/rag/test_embeddings.py
git commit -m "feat(qa): embed_document (RETRIEVAL_DOCUMENT) for qa write path

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `QAIndexer` (Qdrant write path)

**Files:**
- Create: `agent_customer_support/rag/qa_indexer.py`
- Test: `tests/rag/test_qa_indexer.py` (create)

**Interfaces:**
- Consumes: `embed_document` (Task 3); `_normalize_collection` from `agent_customer_support.rag_client`; `QARecord` (Task 1); `Settings.qa_collection`, `Settings.embedding_dim`, `qdrant_endpoint`, `qdrant_api_key`.
- Produces: `QAIndexer(client: AsyncQdrantClient | None = None)` with `async ensure_collection()`, `async upsert(record: QARecord)`, `async delete(point_id: str)`.

- [ ] **Step 1: Write the failing test**

Create `tests/rag/test_qa_indexer.py`:

```python
import pytest
from qdrant_client import AsyncQdrantClient

from agent_customer_support.models import QARecord
from agent_customer_support.rag.qa_indexer import QAIndexer
import agent_customer_support.rag.qa_indexer as qa_indexer_mod
from agent_customer_support.rag_client import _normalize_collection
from agent_customer_support.config import get_settings

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _stub_embed(monkeypatch):
    captured = {}

    async def fake_embed_document(text):
        captured["text"] = text
        return [1.0, 0.0, 0.0]

    monkeypatch.setattr(qa_indexer_mod, "embed_document", fake_embed_document)
    monkeypatch.setattr(get_settings(), "embedding_dim", 3, raising=False)
    return captured


async def test_upsert_embeds_question_and_stores_answer(_stub_embed):
    client = AsyncQdrantClient(location=":memory:")
    idx = QAIndexer(client=client)
    rec = QARecord(question="đổi mật khẩu?", answer="Vào Cài đặt", source="manual",
                   application="Lab")
    await idx.upsert(rec)

    # the vector came from the QUESTION
    assert _stub_embed["text"] == "đổi mật khẩu?"

    physical = _normalize_collection(get_settings().qa_collection)
    got = await client.retrieve(physical, ids=[rec.id], with_payload=True)
    assert len(got) == 1
    payload = got[0].payload
    assert payload["page_content"] == "Vào Cài đặt"           # answer is the content
    assert payload["metadata"]["doc_type"] == "qa"
    assert payload["metadata"]["source_doc_id"] == rec.id
    assert payload["metadata"]["application"] == "Lab"


async def test_delete_removes_point(_stub_embed):
    client = AsyncQdrantClient(location=":memory:")
    idx = QAIndexer(client=client)
    rec = QARecord(question="q", answer="a", source="manual")
    await idx.upsert(rec)
    await idx.delete(rec.id)
    physical = _normalize_collection(get_settings().qa_collection)
    got = await client.retrieve(physical, ids=[rec.id], with_payload=True)
    assert got == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/rag/test_qa_indexer.py -v`
Expected: FAIL (`ModuleNotFoundError: ...rag.qa_indexer`).

- [ ] **Step 3: Implement the indexer**

Create `agent_customer_support/rag/qa_indexer.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/rag/test_qa_indexer.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add agent_customer_support/rag/qa_indexer.py tests/rag/test_qa_indexer.py
git commit -m "feat(qa): QAIndexer upsert/delete into Qdrant qa collection

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Capture A — `KnowledgeAgent` logs a pending Q&A on a confirmed miss

**Files:**
- Modify: `agent_customer_support/agents/context.py` (add `qa_store` handle)
- Modify: `agent_customer_support/agents/coordinator.py` (construct + wire `QAStore`)
- Modify: `agent_customer_support/agents/knowledge.py` (create QARecord on miss)
- Test: `tests/agents/test_knowledge_qa_capture.py` (create)

**Interfaces:**
- Consumes: `QAStore` (Task 2), `QARecord` (Task 1), `TurnContext`.
- Produces: `TurnContext.qa_store` handle; a pending `cannot_answer` `QARecord` created at the existing miss site.

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_knowledge_qa_capture.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_customer_support.agents.knowledge import KnowledgeAgent
from agent_customer_support.agents.context import TurnContext
from agent_customer_support.models import CustomerProfile, SessionState, Conversation

pytestmark = pytest.mark.asyncio


async def test_confirmed_miss_creates_pending_qa_record(monkeypatch):
    agent = KnowledgeAgent()

    # Force the "second miss after clarification" path: compose returns a miss marker.
    monkeypatch.setattr(agent, "_contextualize", AsyncMock(return_value="câu hỏi lạ"))
    monkeypatch.setattr(agent, "_compose", AsyncMock(return_value="[[no_answer]]"))

    session = SessionState(conversation_id="c1", pending="knowledge_clarify")
    ctx = TurnContext(
        customer=CustomerProfile(customer_id="cust1", name="N"),
        session=session,
        conversation=Conversation(conversation_id="c1", customer_id="cust1"),
        message="câu hỏi lạ",
        transcript="assistant: ...\nuser: câu hỏi lạ",
    )
    ctx.rag = MagicMock()
    ctx.rag.search = AsyncMock(return_value={"passages": [], "citations": []})
    ctx.backlog = MagicMock()
    ctx.backlog.add = AsyncMock()
    ctx.qa_store = MagicMock()
    ctx.qa_store.add = AsyncMock()

    res = await agent.run(ctx)

    assert res.resolved is False
    ctx.qa_store.add.assert_awaited_once()
    rec = ctx.qa_store.add.await_args.args[0]
    assert rec.source == "cannot_answer"
    assert rec.status == "pending"
    assert rec.question == "câu hỏi lạ"
    assert rec.conversation_id == "c1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/agents/test_knowledge_qa_capture.py -v`
Expected: FAIL (`AttributeError: 'TurnContext' object has no attribute 'qa_store'` or `qa_store.add` not called).

- [ ] **Step 3: Add the `qa_store` handle to `TurnContext`**

In `agent_customer_support/agents/context.py`, add to the service-handles block:

```python
    backlog: Any = None
    qa_store: Any = None
    escalator: Any = None
```

- [ ] **Step 4: Wire `QAStore` in the Coordinator**

In `agent_customer_support/agents/coordinator.py`: add the import near the other store imports:

```python
from agent_customer_support.stores.qa_store import QAStore
```

In `__init__`, after `self.backlog = RequestBacklog()`:

```python
        self.qa_store = QAStore()
```

In the `TurnContext(...)` construction (where `backlog=self.backlog` appears), add:

```python
                backlog=self.backlog,
                qa_store=self.qa_store,
```

- [ ] **Step 5: Create the QA record at the miss site**

In `agent_customer_support/agents/knowledge.py`, add the import near the top (with the other model import):

```python
from agent_customer_support.models import AgentResult, QARecord
```

At the confirmed-miss site (immediately after the existing `await ctx.backlog.add(...)` call that logs `type="how_to_missing"`, before the final `return AgentResult(...)`), add:

```python
        await ctx.qa_store.add(
            QARecord(
                question=ctx.message,
                source="cannot_answer",
                customer_id=ctx.customer.customer_id,
                conversation_id=ctx.session.conversation_id,
                transcript=ctx.transcript,
            )
        )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `poetry run pytest tests/agents/test_knowledge_qa_capture.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add agent_customer_support/agents/context.py agent_customer_support/agents/coordinator.py agent_customer_support/agents/knowledge.py tests/agents/test_knowledge_qa_capture.py
git commit -m "feat(qa): capture pending Q&A on confirmed knowledge miss

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Channel DI providers + Capture B (feedback endpoint + message_id)

**Files:**
- Create: `agent_customer_support/channels/deps.py`
- Modify: `agent_customer_support/agents/coordinator.py` (`_finish` sets `message_id`)
- Modify: `agent_customer_support/channels/widget.py` (add `/feedback`)
- Test: `tests/channels/test_feedback.py` (create)

**Interfaces:**
- Consumes: `QAStore` (Task 2), `ConversationStore`, `QARecord`, `Conversation`.
- Produces: `channels/deps.py` providers `get_qa_store()`, `get_conversation_store()`, `get_qa_indexer()`, `require_admin(...)`; `POST /widget/feedback`; `ChatResponse.message_id` populated by `_finish`.

- [ ] **Step 1: Write the failing test**

Create `tests/channels/test_feedback.py`:

```python
import pytest
from fastapi.testclient import TestClient

from agent_customer_support.server import app
from agent_customer_support.channels.deps import get_qa_store, get_conversation_store
from agent_customer_support.models import Conversation, Turn

pytestmark = pytest.mark.asyncio


class FakeConvStore:
    def __init__(self, conv):
        self._conv = conv

    async def load(self, conversation_id):
        return self._conv


class FakeQAStore:
    def __init__(self):
        self.records = []

    async def find_by_feedback_message_id(self, mid):
        for r in self.records:
            if r.feedback_message_id == mid:
                return r
        return None

    async def add(self, record):
        self.records.append(record)
        return record

    async def update(self, record):
        return record


def _make_conv():
    user = Turn(role="user", content="Làm sao xoá mẫu?")
    asst = Turn(role="assistant", content="Sai rồi: bạn không thể xoá.")
    return Conversation(conversation_id="c1", customer_id="cust1", turns=[user, asst]), asst.id


def test_feedback_down_creates_record():
    conv, asst_id = _make_conv()
    qa = FakeQAStore()
    app.dependency_overrides[get_conversation_store] = lambda: FakeConvStore(conv)
    app.dependency_overrides[get_qa_store] = lambda: qa
    client = TestClient(app)
    resp = client.post(
        "/widget/feedback",
        json={"conversation_id": "c1", "message_id": asst_id, "signal": "down"},
    )
    assert resp.status_code == 200
    assert len(qa.records) == 1
    rec = qa.records[0]
    assert rec.source == "feedback"
    assert rec.question == "Làm sao xoá mẫu?"
    assert rec.bad_answer == "Sai rồi: bạn không thể xoá."
    assert rec.feedback_message_id == asst_id
    app.dependency_overrides.clear()


def test_feedback_unknown_message_id_404():
    conv, _ = _make_conv()
    app.dependency_overrides[get_conversation_store] = lambda: FakeConvStore(conv)
    app.dependency_overrides[get_qa_store] = lambda: FakeQAStore()
    client = TestClient(app)
    resp = client.post(
        "/widget/feedback",
        json={"conversation_id": "c1", "message_id": "nope", "signal": "down"},
    )
    assert resp.status_code == 404
    app.dependency_overrides.clear()


def test_feedback_repeat_does_not_duplicate():
    conv, asst_id = _make_conv()
    qa = FakeQAStore()
    app.dependency_overrides[get_conversation_store] = lambda: FakeConvStore(conv)
    app.dependency_overrides[get_qa_store] = lambda: qa
    client = TestClient(app)
    body = {"conversation_id": "c1", "message_id": asst_id, "signal": "down"}
    client.post("/widget/feedback", json=body)
    client.post("/widget/feedback", json=body)
    assert len(qa.records) == 1
    app.dependency_overrides.clear()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/channels/test_feedback.py -v`
Expected: FAIL (`ImportError` on `channels.deps`, or 404 route not found).

- [ ] **Step 3: Create the DI providers**

Create `agent_customer_support/channels/deps.py`:

```python
from functools import lru_cache

from fastapi import Header, HTTPException

from agent_customer_support.config import get_settings
from agent_customer_support.rag.qa_indexer import QAIndexer
from agent_customer_support.stores.conversation_store import ConversationStore
from agent_customer_support.stores.qa_store import QAStore


@lru_cache
def get_qa_store() -> QAStore:
    return QAStore()


@lru_cache
def get_conversation_store() -> ConversationStore:
    return ConversationStore()


@lru_cache
def get_qa_indexer() -> QAIndexer:
    return QAIndexer()


def require_admin(x_admin_token: str = Header(default="")) -> None:
    token = get_settings().admin_token
    if not token or x_admin_token != token:
        raise HTTPException(status_code=401, detail="invalid admin token")
```

- [ ] **Step 4: Populate `message_id` in `_finish`**

In `agent_customer_support/agents/coordinator.py`, change the assistant-turn append + return in `_finish` so the assistant `Turn` is built first and its id is returned:

```python
        assistant_turn = Turn(role="assistant", content=result.reply)
        await self.conversations.append(
            ctx.session.conversation_id,
            ctx.customer.customer_id,
            assistant_turn,
        )
        return ChatResponse(
            conversation_id=ctx.session.conversation_id,
            reply=result.reply,
            escalated=result.escalated,
            citations=result.citations,
            message_id=assistant_turn.id,
        )
```

- [ ] **Step 5: Add the feedback endpoint**

In `agent_customer_support/channels/widget.py`, add imports:

```python
from typing import Literal

from fastapi import HTTPException

from agent_customer_support.channels.deps import get_conversation_store, get_qa_store
from agent_customer_support.models import Conversation, QARecord
from agent_customer_support.stores.conversation_store import ConversationStore
from agent_customer_support.stores.qa_store import QAStore
```

Add the request model and route at the end of the file:

```python
class FeedbackRequest(BaseModel):
    conversation_id: str
    message_id: str
    signal: Literal["down"] = "down"


def _transcript(conv: Conversation) -> str:
    return "\n".join(f"{t.role}: {t.content}" for t in conv.turns)


@router.post("/feedback")
async def feedback(
    req: FeedbackRequest,
    qa: QAStore = Depends(get_qa_store),
    convs: ConversationStore = Depends(get_conversation_store),
) -> dict:
    conv = await convs.load(req.conversation_id)
    idx = next(
        (i for i, t in enumerate(conv.turns) if t.id == req.message_id and t.role == "assistant"),
        None,
    )
    if idx is None:
        raise HTTPException(status_code=404, detail="message not found")
    bad_answer = conv.turns[idx].content
    question = next(
        (conv.turns[j].content for j in range(idx - 1, -1, -1) if conv.turns[j].role == "user"),
        "",
    )
    existing = await qa.find_by_feedback_message_id(req.message_id)
    if existing:
        existing.question = question
        existing.bad_answer = bad_answer
        await qa.update(existing)
        return {"ok": True}
    await qa.add(
        QARecord(
            question=question,
            source="feedback",
            status="pending",
            bad_answer=bad_answer,
            customer_id=conv.customer_id or None,
            conversation_id=req.conversation_id,
            feedback_message_id=req.message_id,
            transcript=_transcript(conv),
        )
    )
    return {"ok": True}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `poetry run pytest tests/channels/test_feedback.py -v`
Expected: PASS (3 tests).

- [ ] **Step 7: Commit**

```bash
git add agent_customer_support/channels/deps.py agent_customer_support/channels/widget.py agent_customer_support/agents/coordinator.py tests/channels/test_feedback.py
git commit -m "feat(qa): 👎 feedback endpoint + message_id plumbing

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Admin API (`/admin/qa`)

**Files:**
- Create: `agent_customer_support/channels/admin.py`
- Modify: `agent_customer_support/server.py` (mount router + init `QAStore` in lifespan)
- Test: `tests/channels/test_admin_qa.py` (create)

**Interfaces:**
- Consumes: `get_qa_store`, `get_qa_indexer`, `require_admin` (Task 6); `QARecord`.
- Produces: `admin_router` with list/get/create/patch/approve/reject/archive.

- [ ] **Step 1: Write the failing test**

Create `tests/channels/test_admin_qa.py`:

```python
import pytest
from fastapi.testclient import TestClient

from agent_customer_support.server import app
from agent_customer_support.channels.deps import get_qa_store, get_qa_indexer
from agent_customer_support.models import QARecord

pytestmark = pytest.mark.asyncio

HEADERS = {"X-Admin-Token": "test-admin-token"}


class FakeQAStore:
    def __init__(self):
        self.by_id = {}

    async def add(self, record):
        self.by_id[record.id] = record
        return record

    async def get(self, rid):
        return self.by_id.get(rid)

    async def list(self, status=None):
        return [r for r in self.by_id.values() if status is None or r.status == status]

    async def update(self, record):
        self.by_id[record.id] = record
        return record

    async def delete(self, rid):
        self.by_id.pop(rid, None)


class FakeIndexer:
    def __init__(self):
        self.upserted = []
        self.deleted = []

    async def upsert(self, record):
        self.upserted.append(record.id)

    async def delete(self, point_id):
        self.deleted.append(point_id)


@pytest.fixture
def wired():
    store, indexer = FakeQAStore(), FakeIndexer()
    app.dependency_overrides[get_qa_store] = lambda: store
    app.dependency_overrides[get_qa_indexer] = lambda: indexer
    yield store, indexer, TestClient(app)
    app.dependency_overrides.clear()


def test_requires_admin_token(wired):
    _, _, client = wired
    assert client.get("/admin/qa").status_code == 401


def test_create_list_and_approve_indexes(wired):
    store, indexer, client = wired
    r = client.post("/admin/qa", json={"question": "q1", "answer": "a1"}, headers=HEADERS)
    assert r.status_code == 200
    rid = r.json()["id"]
    assert client.get("/admin/qa?status=pending", headers=HEADERS).json()[0]["id"] == rid

    appr = client.post(f"/admin/qa/{rid}/approve", json={}, headers=HEADERS)
    assert appr.status_code == 200
    assert appr.json()["status"] == "approved"
    assert indexer.upserted == [rid]


def test_approve_empty_answer_409(wired):
    store, indexer, client = wired
    r = client.post("/admin/qa", json={"question": "q-no-answer"}, headers=HEADERS)
    rid = r.json()["id"]
    appr = client.post(f"/admin/qa/{rid}/approve", json={}, headers=HEADERS)
    assert appr.status_code == 409
    assert indexer.upserted == []


def test_archive_deletes_point(wired):
    store, indexer, client = wired
    r = client.post("/admin/qa", json={"question": "q", "answer": "a"}, headers=HEADERS)
    rid = r.json()["id"]
    client.post(f"/admin/qa/{rid}/approve", json={}, headers=HEADERS)
    arch = client.post(f"/admin/qa/{rid}/archive", json={}, headers=HEADERS)
    assert arch.status_code == 200
    assert arch.json()["status"] == "archived"
    assert indexer.deleted == [rid]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/channels/test_admin_qa.py -v`
Expected: FAIL (admin routes 404 / import error).

- [ ] **Step 3: Implement the admin router**

Create `agent_customer_support/channels/admin.py`:

```python
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from agent_customer_support.channels.deps import get_qa_indexer, get_qa_store, require_admin
from agent_customer_support.models import QARecord
from agent_customer_support.rag.qa_indexer import QAIndexer
from agent_customer_support.stores.qa_store import QAStore

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


class QACreate(BaseModel):
    question: str
    answer: str = ""
    application: str | None = None


class QAPatch(BaseModel):
    question: str | None = None
    answer: str | None = None
    application: str | None = None


class ApproveBody(BaseModel):
    approved_by: str | None = None


@router.get("/qa")
async def list_qa(status: str | None = None, qa: QAStore = Depends(get_qa_store)) -> list[QARecord]:
    return await qa.list(status=status)


@router.get("/qa/{record_id}")
async def get_qa(record_id: str, qa: QAStore = Depends(get_qa_store)) -> QARecord:
    rec = await qa.get(record_id)
    if not rec:
        raise HTTPException(status_code=404, detail="not found")
    return rec


@router.post("/qa")
async def create_qa(body: QACreate, qa: QAStore = Depends(get_qa_store)) -> QARecord:
    return await qa.add(
        QARecord(
            question=body.question,
            answer=body.answer,
            application=body.application,
            source="manual",
        )
    )


@router.patch("/qa/{record_id}")
async def edit_qa(
    record_id: str,
    patch: QAPatch,
    qa: QAStore = Depends(get_qa_store),
    indexer: QAIndexer = Depends(get_qa_indexer),
) -> QARecord:
    rec = await qa.get(record_id)
    if not rec:
        raise HTTPException(status_code=404, detail="not found")
    if patch.question is not None:
        rec.question = patch.question
    if patch.answer is not None:
        rec.answer = patch.answer
    if patch.application is not None:
        rec.application = patch.application
    await qa.update(rec)
    if rec.status == "approved":
        await indexer.upsert(rec)
    return rec


@router.post("/qa/{record_id}/approve")
async def approve_qa(
    record_id: str,
    body: ApproveBody,
    qa: QAStore = Depends(get_qa_store),
    indexer: QAIndexer = Depends(get_qa_indexer),
) -> QARecord:
    rec = await qa.get(record_id)
    if not rec:
        raise HTTPException(status_code=404, detail="not found")
    if not rec.answer.strip():
        raise HTTPException(status_code=409, detail="answer required before approval")
    await indexer.upsert(rec)  # index-first: only persist approval if this succeeds
    rec.status = "approved"
    rec.approved_by = body.approved_by
    rec.indexed_at = datetime.now(UTC)
    rec.qdrant_point_id = rec.id
    return await qa.update(rec)


@router.post("/qa/{record_id}/reject")
async def reject_qa(record_id: str, qa: QAStore = Depends(get_qa_store)) -> QARecord:
    rec = await qa.get(record_id)
    if not rec:
        raise HTTPException(status_code=404, detail="not found")
    rec.status = "rejected"
    return await qa.update(rec)


@router.post("/qa/{record_id}/archive")
async def archive_qa(
    record_id: str,
    qa: QAStore = Depends(get_qa_store),
    indexer: QAIndexer = Depends(get_qa_indexer),
) -> QARecord:
    rec = await qa.get(record_id)
    if not rec:
        raise HTTPException(status_code=404, detail="not found")
    await indexer.delete(rec.id)
    rec.status = "archived"
    rec.qdrant_point_id = None
    return await qa.update(rec)
```

- [ ] **Step 4: Mount the router + init the store**

In `agent_customer_support/server.py`:

Add the import next to the widget router import:

```python
from agent_customer_support.channels.admin import router as admin_router
from agent_customer_support.stores.qa_store import QAStore
```

Add `QAStore()` to the lifespan init tuple:

```python
    for store in (CustomerRegistry(), ConversationStore(), FlowStore(), RequestBacklog(), QAStore()):
```

Mount the router after `app.include_router(widget_router)`:

```python
app.include_router(admin_router)
```

Update the CORS `allow_headers` to permit the admin token header:

```python
    allow_headers=["Content-Type", "X-Admin-Token"],
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `poetry run pytest tests/channels/test_admin_qa.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add agent_customer_support/channels/admin.py agent_customer_support/server.py tests/channels/test_admin_qa.py
git commit -m "feat(qa): /admin/qa curation API (token-gated, index-first approve)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: CS admin UI + widget 👎 control

**Files:**
- Create: `ui/app/admin/page.tsx`
- Modify: `ui/lib/api.ts` (admin + feedback calls)
- Modify: `ui/components/MessageList.tsx` (👎 button on assistant messages)
- Modify: `ui/app/page.tsx` (capture `message_id`, pass feedback handler)

**Interfaces:**
- Consumes: `/admin/qa` API (Task 7), `/widget/feedback` (Task 6), `ChatResponse.message_id` (Task 1).
- Produces: a usable CS curation page and a customer-facing 👎 control. (Frontend — verified by build + manual check, not unit tests.)

- [ ] **Step 1: Add API client functions**

In `ui/lib/api.ts`, add (adjust `API_BASE` to the existing constant name in the file):

```ts
export async function sendFeedback(conversationId: string, messageId: string) {
  await fetch(`${API_BASE}/widget/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ conversation_id: conversationId, message_id: messageId, signal: "down" }),
  });
}

function adminHeaders(token: string) {
  return { "Content-Type": "application/json", "X-Admin-Token": token };
}

export async function listQA(token: string, status = "pending") {
  const r = await fetch(`${API_BASE}/admin/qa?status=${status}`, { headers: adminHeaders(token) });
  if (!r.ok) throw new Error(`list failed: ${r.status}`);
  return r.json();
}

export async function approveQA(token: string, id: string) {
  const r = await fetch(`${API_BASE}/admin/qa/${id}/approve`, {
    method: "POST", headers: adminHeaders(token), body: "{}",
  });
  if (!r.ok) throw new Error(`approve failed: ${r.status}`);
  return r.json();
}

export async function rejectQA(token: string, id: string) {
  const r = await fetch(`${API_BASE}/admin/qa/${id}/reject`, {
    method: "POST", headers: adminHeaders(token), body: "{}",
  });
  if (!r.ok) throw new Error(`reject failed: ${r.status}`);
  return r.json();
}

export async function editQA(token: string, id: string, patch: Record<string, unknown>) {
  const r = await fetch(`${API_BASE}/admin/qa/${id}`, {
    method: "PATCH", headers: adminHeaders(token), body: JSON.stringify(patch),
  });
  if (!r.ok) throw new Error(`edit failed: ${r.status}`);
  return r.json();
}
```

- [ ] **Step 2: Build the admin page**

Create `ui/app/admin/page.tsx`:

```tsx
"use client";

import { useEffect, useState } from "react";
import { listQA, approveQA, rejectQA, editQA } from "@/lib/api";

type QA = {
  id: string; question: string; answer: string; status: string;
  source: string; bad_answer?: string | null; transcript?: string;
  application?: string | null;
};

export default function AdminPage() {
  const [token, setToken] = useState("");
  const [items, setItems] = useState<QA[]>([]);
  const [sel, setSel] = useState<QA | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    const t = localStorage.getItem("adminToken") || "";
    setToken(t);
  }, []);

  async function refresh(t: string) {
    try {
      setError("");
      setItems(await listQA(t, "pending"));
    } catch (e) {
      setError(String(e));
    }
  }

  function saveToken() {
    localStorage.setItem("adminToken", token);
    refresh(token);
  }

  async function onApprove() {
    if (!sel) return;
    await editQA(token, sel.id, { answer: sel.answer, application: sel.application ?? null });
    await approveQA(token, sel.id);
    setSel(null);
    refresh(token);
  }

  async function onReject() {
    if (!sel) return;
    await rejectQA(token, sel.id);
    setSel(null);
    refresh(token);
  }

  return (
    <main style={{ display: "flex", gap: 24, padding: 24 }}>
      <section style={{ width: 360 }}>
        <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
          <input value={token} onChange={(e) => setToken(e.target.value)} placeholder="Admin token"
            style={{ flex: 1 }} />
          <button onClick={saveToken}>Load</button>
        </div>
        {error && <p style={{ color: "red" }}>{error}</p>}
        <h3>Pending ({items.length})</h3>
        <ul style={{ listStyle: "none", padding: 0 }}>
          {items.map((it) => (
            <li key={it.id}>
              <button onClick={() => setSel(it)} style={{ textAlign: "left", width: "100%" }}>
                [{it.source}] {it.question}
              </button>
            </li>
          ))}
        </ul>
      </section>
      <section style={{ flex: 1 }}>
        {sel ? (
          <div>
            <h3>{sel.question}</h3>
            {sel.bad_answer && (
              <p><b>Câu trả lời sai:</b> {sel.bad_answer}</p>
            )}
            {sel.transcript && (
              <details><summary>Transcript</summary><pre>{sel.transcript}</pre></details>
            )}
            <label>Câu trả lời đúng</label>
            <textarea
              value={sel.answer}
              onChange={(e) => setSel({ ...sel, answer: e.target.value })}
              rows={8} style={{ width: "100%" }}
            />
            <input
              value={sel.application ?? ""}
              onChange={(e) => setSel({ ...sel, application: e.target.value })}
              placeholder="application (optional)" style={{ width: "100%", marginTop: 8 }}
            />
            <div style={{ marginTop: 12, display: "flex", gap: 8 }}>
              <button onClick={onApprove} disabled={!sel.answer.trim()}>Approve</button>
              <button onClick={onReject}>Reject</button>
            </div>
          </div>
        ) : (
          <p>Chọn một câu hỏi để xử lý.</p>
        )}
      </section>
    </main>
  );
}
```

- [ ] **Step 3: Add the 👎 control to the chat**

In `ui/app/page.tsx`, capture `message_id` from the chat response into the assistant message object (find where the assistant reply is appended after calling the chat API, and store `message_id` from the response alongside the text). Then in `ui/components/MessageList.tsx`, for each assistant message that has a `messageId`, render a small 👎 button:

```tsx
{m.role === "assistant" && m.messageId && (
  <button
    aria-label="Không hữu ích"
    onClick={() => onFeedbackDown(m.messageId!)}
    style={{ marginLeft: 8, opacity: 0.6 }}
  >
    👎
  </button>
)}
```

Wire `onFeedbackDown` from `page.tsx` to call `sendFeedback(conversationId, messageId)` (import from `@/lib/api`), and add `messageId?: string` to the message type used by `MessageList`. (Match the existing prop-passing style in these files.)

- [ ] **Step 4: Verify the UI builds**

Run: `cd ui && npm run build`
Expected: build succeeds with no type errors in the new/changed files.

- [ ] **Step 5: Manual verification (document, do not block on services)**

With the API running (`make run`, `ADMIN_TOKEN` set) and `cd ui && npm run dev`:
1. Send a chat message; confirm a 👎 appears on the assistant reply and clicking it returns 200 (Network tab).
2. Open `/admin`, enter the admin token, see the pending item, write an answer, Approve; confirm it disappears from pending.

- [ ] **Step 6: Commit**

```bash
git add ui/app/admin/page.tsx ui/lib/api.ts ui/components/MessageList.tsx ui/app/page.tsx
git commit -m "feat(qa): CS admin curation page + widget 👎 feedback control

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- `QARecord` model + lifecycle → Task 1 (model), Tasks 5/6/7 (transitions). ✓
- `QAStore` (acs_qa, CRUD, status list, feedback lookup) → Task 2. ✓
- Capture A (cannot_answer in KnowledgeAgent, keeps backlog) → Task 5. ✓
- Capture B (feedback endpoint, message_id plumbing, dedup repeat) → Task 6. ✓
- Capture C (manual create) → Task 7 (`POST /admin/qa`). ✓
- Write path: `embed_document` → Task 3; `QAIndexer` (question-only vector, answer in page_content, `_v3` name, ensure_collection) → Task 4. ✓
- CS API (token gate, list/get/create/patch/approve/reject/archive, index-first approve, 409 empty answer) → Task 7. ✓
- CS UI (`/admin` page, token in localStorage, 👎 control) → Task 8. ✓
- Config (`table_qa`, `qa_collection`, `admin_token`) → Task 1. ✓
- Error handling (401/404/409, index-first 502 semantics) → Tasks 6/7 (approve upsert raises → 502 surfaces via FastAPI default 500/502 from QAIndexer error; index-first ordering guarantees no half-approval). ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases"; every code step has complete code. Task 8 step 3 references existing prop-passing patterns rather than rewriting unseen files verbatim — intentional, since `page.tsx`/`MessageList.tsx` shapes are app-specific; the executor must match the current structure. ✓

**Type consistency:** `QARecord` fields are identical across Tasks 1/2/4/5/6/7. `QAStore` method names (`add/get/list/update/delete/find_by_feedback_message_id`) match between Task 2 definition and the fakes/uses in Tasks 6/7. `QAIndexer.upsert/delete/ensure_collection` match between Task 4 and Task 7. `get_qa_store/get_qa_indexer/get_conversation_store/require_admin` defined in Task 6, used in Tasks 6/7. `_normalize_collection` reused from `rag_client` (Spec 1). ✓

**Note for executor:** Tasks 2 and 7's store tests need DynamoDB Local (`make infra-up`); Task 4 uses in-memory Qdrant (no infra). Task 8 is frontend — there are pre-existing unrelated WIP test failures in the repo (rearchitecture in progress); only the new QA tests must pass.

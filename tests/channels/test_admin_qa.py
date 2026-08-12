import pytest
from fastapi.testclient import TestClient

from agent_customer_support.server import app
from agent_customer_support.channels.deps import get_qa_store, get_qa_indexer

pytestmark = pytest.mark.asyncio


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


def test_requires_authentication(wired):
    _, _, client = wired
    assert client.get("/admin/qa").status_code == 401


def test_non_admin_role_is_forbidden(wired, as_user):
    _, _, client = wired
    assert client.get("/admin/qa").status_code == 403


def test_create_list_and_approve_indexes(wired, as_admin):
    store, indexer, client = wired
    r = client.post("/admin/qa", json={"question": "q1", "answer": "a1"})
    assert r.status_code == 200
    rid = r.json()["id"]
    assert client.get("/admin/qa?status=pending").json()[0]["id"] == rid

    appr = client.post(f"/admin/qa/{rid}/approve", json={})
    assert appr.status_code == 200
    assert appr.json()["status"] == "approved"
    assert indexer.upserted == [rid]


def test_approve_empty_answer_409(wired, as_admin):
    store, indexer, client = wired
    r = client.post("/admin/qa", json={"question": "q-no-answer"})
    rid = r.json()["id"]
    appr = client.post(f"/admin/qa/{rid}/approve", json={})
    assert appr.status_code == 409
    assert indexer.upserted == []


def test_approve_indexing_failure_returns_502(wired, as_admin):
    store, _, client = wired

    class FailingIndexer(FakeIndexer):
        async def upsert(self, record):
            raise RuntimeError("qdrant down")

    failing_indexer = FailingIndexer()
    app.dependency_overrides[get_qa_indexer] = lambda: failing_indexer

    r = client.post("/admin/qa", json={"question": "q", "answer": "a"})
    rid = r.json()["id"]

    appr = client.post(f"/admin/qa/{rid}/approve", json={})
    assert appr.status_code == 502
    assert store.by_id[rid].status != "approved"


def test_archive_deletes_point(wired, as_admin):
    store, indexer, client = wired
    r = client.post("/admin/qa", json={"question": "q", "answer": "a"})
    rid = r.json()["id"]
    client.post(f"/admin/qa/{rid}/approve", json={})
    arch = client.post(f"/admin/qa/{rid}/archive", json={})
    assert arch.status_code == 200
    assert arch.json()["status"] == "archived"
    assert indexer.deleted == [rid]

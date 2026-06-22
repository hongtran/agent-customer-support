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

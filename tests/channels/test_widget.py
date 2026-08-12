from fastapi.testclient import TestClient
from unittest.mock import AsyncMock
from agent_customer_support.models import ChatResponse
from agent_customer_support.server import app, get_agent


def test_chat_endpoint_returns_reply(as_user):
    fake = AsyncMock()
    fake.handle_turn.return_value = ChatResponse(
        conversation_id="cv1", reply="Xin chào", citations=["c#1"]
    )
    app.dependency_overrides[get_agent] = lambda: fake
    client = TestClient(app)
    resp = client.post(
        "/widget/chat",
        json={"conversation_id": "cv1", "message": "hi"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["reply"] == "Xin chào" and body["conversation_id"] == "cv1"
    app.dependency_overrides.clear()


def test_chat_passes_attachments(as_user):
    fake = AsyncMock()
    fake.handle_turn.return_value = ChatResponse(conversation_id="cv1", reply="hi")
    app.dependency_overrides[get_agent] = lambda: fake
    client = TestClient(app)
    resp = client.post(
        "/widget/chat",
        json={
            "conversation_id": "cv1",
            "message": "hello",
            "attachments": [{"kind": "image", "media_type": "image/png", "data": "QUJD"}],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["reply"] == "hi"
    kwargs = fake.handle_turn.call_args.kwargs
    assert kwargs["message"] == "hello"
    assert len(kwargs["attachments"]) == 1
    app.dependency_overrides.clear()


def test_health():
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}


def test_get_agent_is_singleton():
    from agent_customer_support.channels.widget import get_agent

    assert get_agent() is get_agent()

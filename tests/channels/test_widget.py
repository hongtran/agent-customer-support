from fastapi.testclient import TestClient
from unittest.mock import AsyncMock
from agent_customer_support.models import ChatResponse
from agent_customer_support.server import app, get_agent

def test_chat_endpoint_returns_reply():
    fake = AsyncMock()
    fake.handle_turn.return_value = ChatResponse(conversation_id="cv1", reply="Xin chào", citations=["c#1"])
    app.dependency_overrides[get_agent] = lambda: fake
    client = TestClient(app)
    resp = client.post("/widget/chat", json={"customer_id": "c1", "conversation_id": "cv1", "message": "hi"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["reply"] == "Xin chào" and body["conversation_id"] == "cv1"
    app.dependency_overrides.clear()

def test_health():
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}

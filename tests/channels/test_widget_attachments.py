import base64

import pytest
from fastapi.testclient import TestClient

from agent_customer_support.config import get_settings
from agent_customer_support.models import ChatResponse
from agent_customer_support.server import app, get_agent

# Deliberately no `pytestmark = pytest.mark.asyncio` here: every test drives the app
# through the sync TestClient, and marking sync tests as asyncio only emits warnings.

PNG = "image/png"


def _b64(n: int) -> str:
    return base64.b64encode(b"\0" * n).decode()


def _payload(attachments=None, message="cái này lỗi gì?"):
    body = {
        "conversation_id": "conv1",
        "message": message,
    }
    if attachments is not None:
        body["attachments"] = attachments
    return body


class SpyCoordinator:
    """Records whether the agent was reached at all — rejected uploads must never
    get far enough to cost an LLM call."""

    def __init__(self):
        self.calls = 0

    async def handle_turn(self, **kwargs):
        self.calls += 1
        self.seen = kwargs
        return ChatResponse(conversation_id="conv1", reply="ok", message_id="m1")


@pytest.fixture
def spy():
    s = SpyCoordinator()
    app.dependency_overrides[get_agent] = lambda: s
    yield s
    app.dependency_overrides.clear()


def test_oversized_attachment_is_rejected_with_413(spy, as_user):
    limit = get_settings().max_attachment_bytes
    with TestClient(app) as c:
        res = c.post(
            "/widget/chat",
            json=_payload([{"kind": "image", "media_type": PNG, "data": _b64(limit + 1024)}]),
        )
    assert res.status_code == 413
    assert "limit" in res.json()["detail"]
    # the whole point of checking at the boundary
    assert spy.calls == 0


def test_several_attachments_are_summed_against_the_limit(spy, as_user):
    limit = get_settings().max_attachment_bytes
    half = limit // 2 + 1024
    with TestClient(app) as c:
        res = c.post(
            "/widget/chat",
            json=_payload(
                [
                    {"kind": "image", "media_type": PNG, "data": _b64(half)},
                    {"kind": "image", "media_type": PNG, "data": _b64(half)},
                ]
            ),
        )
    assert res.status_code == 413
    assert spy.calls == 0


def test_attachment_under_the_limit_is_accepted(spy, as_user):
    # 1 MB: comfortably past the old ~300 KB DynamoDB ceiling, under the new cap
    with TestClient(app) as c:
        res = c.post(
            "/widget/chat",
            json=_payload([{"kind": "image", "media_type": PNG, "data": _b64(1024 * 1024)}]),
        )
    assert res.status_code == 200
    assert spy.calls == 1
    assert len(spy.seen["attachments"]) == 1


def test_malformed_base64_is_rejected_with_422(spy, as_user):
    with TestClient(app) as c:
        res = c.post(
            "/widget/chat",
            json=_payload([{"kind": "image", "media_type": PNG, "data": "not!valid!base64!"}]),
        )
    assert res.status_code == 422
    assert spy.calls == 0


def test_unsupported_media_type_is_rejected_with_422(spy, as_user):
    with TestClient(app) as c:
        res = c.post(
            "/widget/chat",
            json=_payload([{"kind": "image", "media_type": "image/gif", "data": _b64(64)}]),
        )
    assert res.status_code == 422
    assert spy.calls == 0


def test_no_attachments_is_unaffected(spy, as_user):
    with TestClient(app) as c:
        res = c.post("/widget/chat", json=_payload())
    assert res.status_code == 200
    assert spy.calls == 1

"""End-to-end token path: real JWTs, real bcrypt, a fake registry in place of DynamoDB.

These are the tests that prove login and the identity dependency agree with each other —
everything else in tests/channels/ overrides get_current_customer and takes it on faith.
"""

import pytest
from fastapi.testclient import TestClient

from agent_customer_support.auth import hash_password
from agent_customer_support.channels.deps import get_customer_registry
from agent_customer_support.models import ChatResponse, CustomerProfile
from agent_customer_support.server import app, get_agent


class FakeRegistry:
    def __init__(self, *profiles: CustomerProfile):
        self.by_id = {p.customer_id: p for p in profiles}

    async def get(self, customer_id):
        return self.by_id.get(customer_id)

    async def list(self):
        return list(self.by_id.values())


CUSTOMER = CustomerProfile(
    customer_id="ttp",
    name="TTP",
    enabled_applications=["Lấy mẫu - Quan trắc"],
    password_hash=hash_password("secret-password"),
)
NO_PASSWORD = CustomerProfile(customer_id="legacy", name="Legacy")
ADMIN = CustomerProfile(
    customer_id="admin", name="Admin", role="admin", password_hash=hash_password("admin-password")
)


class SpyCoordinator:
    def __init__(self):
        self.seen = None

    async def handle_turn(self, **kwargs):
        self.seen = kwargs
        return ChatResponse(conversation_id=kwargs["conversation_id"], reply="ok")


@pytest.fixture
def client():
    app.dependency_overrides[get_customer_registry] = lambda: FakeRegistry(
        CUSTOMER, NO_PASSWORD, ADMIN
    )
    yield TestClient(app)
    app.dependency_overrides.clear()


def _login(client, user_name, password):
    return client.post("/auth/login", json={"user_name": user_name, "password": password})


def _token(client, user_name="ttp", password="secret-password"):
    return _login(client, user_name, password).json()["access_token"]


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_login_returns_a_token(client):
    r = _login(client, "ttp", "secret-password")
    assert r.status_code == 200
    body = r.json()
    assert body["access_token"] and body["token_type"] == "bearer"
    assert body["customer_id"] == "ttp" and body["role"] == "user"


def test_login_never_returns_the_hash(client):
    assert "password_hash" not in _login(client, "ttp", "secret-password").text


def test_wrong_password_unknown_id_and_no_credentials_are_indistinguishable(client):
    # Identical status AND body: any difference between these three tells an attacker
    # which customer ids exist and which have credentials set.
    wrong = _login(client, "ttp", "nope")
    unknown = _login(client, "nobody", "nope")
    no_pw = _login(client, "legacy", "nope")
    assert wrong.status_code == unknown.status_code == no_pw.status_code == 401
    assert wrong.json() == unknown.json() == no_pw.json()


def test_me_returns_the_token_owner(client):
    r = client.get("/auth/me", headers=_auth(_token(client)))
    assert r.status_code == 200
    assert r.json() == {
        "customer_id": "ttp",
        "name": "TTP",
        "role": "user",
        "enabled_applications": ["Lấy mẫu - Quan trắc"],
    }


def test_me_without_a_token_is_401(client):
    assert client.get("/auth/me").status_code == 401


def test_me_with_a_garbage_token_is_401(client):
    assert client.get("/auth/me", headers=_auth("not.a.jwt")).status_code == 401


def test_token_for_a_deleted_customer_is_rejected(client):
    """The profile is re-read per request, so revocation doesn't wait for expiry."""
    token = _token(client)
    app.dependency_overrides[get_customer_registry] = lambda: FakeRegistry()  # customer gone
    assert client.get("/auth/me", headers=_auth(token)).status_code == 401


def test_chat_requires_a_token(client):
    r = client.post("/widget/chat", json={"conversation_id": "cv1", "message": "hi"})
    assert r.status_code == 401


def test_chat_identity_comes_from_the_token_not_the_body(client):
    spy = SpyCoordinator()
    app.dependency_overrides[get_agent] = lambda: spy
    token = _token(client)
    r = client.post(
        "/widget/chat",
        # A hostile client tries to act as another tenant. The field isn't on
        # ChatRequest any more, so it is ignored rather than trusted.
        json={"customer_id": "someone-else", "conversation_id": "cv1", "message": "hi"},
        headers=_auth(token),
    )
    assert r.status_code == 200
    assert spy.seen["customer_id"] == "ttp"


def test_my_applications_come_from_the_token(client):
    r = client.get("/widget/me/applications", headers=_auth(_token(client)))
    assert r.status_code == 200
    assert r.json() == {"customer_id": "ttp", "enabled_applications": ["Lấy mẫu - Quan trắc"]}


def test_admin_routes_reject_a_user_role_token(client):
    assert client.get("/admin/qa", headers=_auth(_token(client))).status_code == 403


def test_admin_routes_accept_an_admin_role_token(client):
    token = _token(client, "admin", "admin-password")
    # 403 is the thing under test; any other status means the role check passed.
    assert client.get("/admin/customers", headers=_auth(token)).status_code != 403


def test_role_is_read_from_the_profile_not_the_token(client):
    """A token minted while the caller was an admin stops working the moment the
    stored profile says otherwise — the point of re-reading the profile."""
    token = _token(client, "admin", "admin-password")
    demoted = ADMIN.model_copy(update={"role": "user"})
    app.dependency_overrides[get_customer_registry] = lambda: FakeRegistry(demoted)
    assert client.get("/admin/qa", headers=_auth(token)).status_code == 403

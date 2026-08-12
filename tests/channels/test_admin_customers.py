import pytest
from fastapi.testclient import TestClient

from agent_customer_support.auth import hash_password, verify_password
from agent_customer_support.channels.deps import get_customer_registry
from agent_customer_support.models import CustomerProfile
from agent_customer_support.server import app
from agent_customer_support.stores.customer_registry import CustomerExistsError

EXISTING = CustomerProfile(
    customer_id="ttp",
    name="TTP",
    enabled_applications=["Lấy mẫu - Quan trắc"],
    password_hash=hash_password("original-password"),
)


class FakeRegistry:
    """In-memory stand-in that reproduces the one behaviour that matters here:
    create() refuses to clobber an existing customer_id, put() overwrites."""

    def __init__(self, *profiles: CustomerProfile):
        self.by_id = {p.customer_id: p.model_copy(deep=True) for p in profiles}

    async def get(self, customer_id):
        p = self.by_id.get(customer_id)
        return p.model_copy(deep=True) if p else None

    async def list(self):
        return [p.model_copy(deep=True) for p in self.by_id.values()]

    async def create(self, profile):
        if profile.customer_id in self.by_id:
            raise CustomerExistsError(profile.customer_id)
        self.by_id[profile.customer_id] = profile

    async def put(self, profile):
        self.by_id[profile.customer_id] = profile


@pytest.fixture
def reg(as_admin):
    r = FakeRegistry(EXISTING)
    app.dependency_overrides[get_customer_registry] = lambda: r
    yield r
    app.dependency_overrides.clear()


@pytest.fixture
def client(reg):
    return TestClient(app)


def _create(client, **overrides):
    body = {"customer_id": "newco", "name": "New Co", "password": "new-password"} | overrides
    return client.post("/admin/customers", json=body)


def test_create_returns_201_and_stores_a_hash(client, reg):
    r = _create(client)
    assert r.status_code == 201
    assert r.json()["customer_id"] == "newco"
    assert r.json()["has_password"] is True
    stored = reg.by_id["newco"]
    # The plaintext must never reach the store.
    assert stored.password_hash != "new-password"
    assert verify_password("new-password", stored.password_hash)


def test_create_never_returns_the_hash(client, reg):
    assert "password_hash" not in _create(client).text


def test_list_never_returns_hashes(client, reg):
    assert "password_hash" not in client.get("/admin/customers").text


def test_duplicate_customer_id_is_409_and_leaves_the_original_intact(client, reg):
    before = reg.by_id["ttp"].model_copy(deep=True)
    r = _create(client, customer_id="ttp", name="Impostor", password="stolen-password")
    assert r.status_code == 409
    after = reg.by_id["ttp"]
    # The dangerous failure mode: an upsert would have handed a live tenant's id,
    # conversations and application scope to whoever submitted this form.
    assert after.name == before.name
    assert after.password_hash == before.password_hash
    assert after.enabled_applications == before.enabled_applications


def test_create_rejects_a_malformed_customer_id(client, reg):
    assert _create(client, customer_id="has spaces").status_code == 422
    assert _create(client, customer_id="").status_code == 422


def test_create_rejects_a_short_password(client, reg):
    assert _create(client, password="short").status_code == 422


def test_patch_without_a_password_leaves_the_hash_alone(client, reg):
    before = reg.by_id["ttp"].password_hash
    r = client.patch("/admin/customers/ttp", json={"name": "TTP Renamed"})
    assert r.status_code == 200
    assert r.json()["name"] == "TTP Renamed"
    assert reg.by_id["ttp"].password_hash == before


def test_patch_with_a_password_replaces_it(client, reg):
    r = client.patch("/admin/customers/ttp", json={"password": "brand-new-password"})
    assert r.status_code == 200
    stored = reg.by_id["ttp"].password_hash
    assert verify_password("brand-new-password", stored)
    assert not verify_password("original-password", stored)


def test_patch_can_change_role_and_applications(client, reg):
    r = client.patch(
        "/admin/customers/ttp",
        json={"role": "admin", "enabled_applications": ["Phòng thí nghiệm"]},
    )
    assert r.status_code == 200
    assert reg.by_id["ttp"].role == "admin"
    assert reg.by_id["ttp"].enabled_applications == ["Phòng thí nghiệm"]


def test_patch_unknown_customer_is_404(client, reg):
    assert client.patch("/admin/customers/nobody", json={"name": "x"}).status_code == 404


def test_created_customer_can_log_in(client, reg):
    """The round trip that proves creation and login agree on the hashing scheme."""
    _create(client)
    r = client.post("/auth/login", json={"user_name": "newco", "password": "new-password"})
    assert r.status_code == 200
    assert r.json()["customer_id"] == "newco"


def test_customer_routes_require_admin_role(as_user):
    r = FakeRegistry(EXISTING)
    app.dependency_overrides[get_customer_registry] = lambda: r
    c = TestClient(app)
    assert c.get("/admin/customers").status_code == 403
    assert (
        c.post(
            "/admin/customers", json={"customer_id": "x", "name": "X", "password": "pw-long-enough"}
        ).status_code
        == 403
    )
    assert "x" not in r.by_id
    app.dependency_overrides.clear()

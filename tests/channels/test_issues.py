import pytest
from fastapi.testclient import TestClient

from agent_customer_support.channels.deps import (
    get_customer_registry,
    get_doc_image_store,
    get_request_backlog,
)
from agent_customer_support.models import CustomerProfile, IssueSummary, RequestRecord
from agent_customer_support.server import app


class FakeBacklog:
    def __init__(self, *rows: RequestRecord):
        self.rows = {r.id: r for r in rows}
        self.calls: list[dict] = []

    async def get(self, request_id):
        r = self.rows.get(request_id)
        return r.model_copy(deep=True) if r else None

    async def list_bugs(self, customer_id):
        rows = [r for r in self.rows.values() if r.customer_id == customer_id and r.type == "bug"]
        rows.sort(key=lambda r: r.created_at, reverse=True)
        return [IssueSummary.model_validate(r.model_dump()) for r in rows]

    async def update_issue(self, request_id, customer_id, *, by, **changes):
        r = self.rows.get(request_id)
        if r is None or r.customer_id != customer_id or r.type != "bug":
            return None
        self.calls.append(changes)
        if "status" in changes:
            r.status = changes["status"]
            r.status_updated_by = by
        if "note" in changes:
            r.note = changes["note"]
        if "due_date" in changes:
            r.due_date = changes["due_date"]
        return r.model_copy(deep=True)


class FakeRegistry:
    async def get(self, customer_id):
        return (
            CustomerProfile(customer_id=customer_id, name="X") if customer_id != "ghost" else None
        )


class FakeDocImages:
    async def presign(self, slug, name):
        return f"https://signed/{slug}/{name}"


MINE = RequestRecord(
    id="b1",
    customer_id="cust1",
    type="bug",
    summary="Import báo lỗi 500",
    title="Import lỗi",
    transcript="user: lỗi\nassistant: xem [[img:screen:lay_mau_quan_trac/image1.png]]",
)
THEIRS = RequestRecord(id="b2", customer_id="other", type="bug", summary="của người khác")
HOWTO = RequestRecord(id="h1", customer_id="cust1", type="how_to_missing", summary="hỏi")


@pytest.fixture
def backlog():
    b = FakeBacklog(MINE.model_copy(deep=True), THEIRS, HOWTO)
    app.dependency_overrides[get_request_backlog] = lambda: b
    app.dependency_overrides[get_customer_registry] = lambda: FakeRegistry()
    app.dependency_overrides[get_doc_image_store] = lambda: FakeDocImages()
    yield b
    for dep in (get_request_backlog, get_customer_registry, get_doc_image_store):
        app.dependency_overrides.pop(dep, None)


# ---- customer ----


def test_user_lists_only_own_bugs(as_user, backlog):
    r = TestClient(app).get("/widget/me/issues")
    assert r.status_code == 200
    assert [i["id"] for i in r.json()] == ["b1"]
    assert r.json()[0]["status"] == "pending"
    assert "transcript" not in r.json()[0]


def test_user_detail_has_summary_and_resolved_transcript(as_user, backlog):
    r = TestClient(app).get("/widget/me/issues/b1")
    assert r.status_code == 200
    body = r.json()
    assert body["summary"] == "Import báo lỗi 500"
    assert "[[img:" not in body["transcript"]
    assert "https://signed/lay_mau_quan_trac/image1.png" in body["transcript"]


@pytest.mark.parametrize("issue_id", ["b2", "h1", "missing"])
def test_user_cannot_open_other_customers_or_non_bug_rows(as_user, backlog, issue_id):
    assert TestClient(app).get(f"/widget/me/issues/{issue_id}").status_code == 404


# ---- admin ----


def test_admin_lists_one_customers_bugs(as_admin, backlog):
    r = TestClient(app).get("/admin/customers/cust1/issues")
    assert r.status_code == 200
    assert [i["id"] for i in r.json()] == ["b1"]


def test_admin_unknown_customer_is_404(as_admin, backlog):
    assert TestClient(app).get("/admin/customers/ghost/issues").status_code == 404


def test_admin_cannot_open_an_issue_under_the_wrong_customer(as_admin, backlog):
    c = TestClient(app)
    assert c.get("/admin/customers/cust1/issues/b2").status_code == 404
    assert (
        c.patch("/admin/customers/cust1/issues/b2", json={"status": "completed"}).status_code == 404
    )


def test_admin_updates_status(as_admin, backlog):
    r = TestClient(app).patch("/admin/customers/cust1/issues/b1", json={"status": "processing"})
    assert r.status_code == 200
    assert r.json()["status"] == "processing"
    assert backlog.rows["b1"].status_updated_by == "admin"


def test_admin_bad_status_is_422(as_admin, backlog):
    r = TestClient(app).patch("/admin/customers/cust1/issues/b1", json={"status": "done"})
    assert r.status_code == 422


def test_admin_unknown_issue_is_404(as_admin, backlog):
    r = TestClient(app).patch("/admin/customers/cust1/issues/nope", json={"status": "completed"})
    assert r.status_code == 404


def test_non_admin_cannot_use_admin_routes(as_user, backlog):
    c = TestClient(app)
    assert c.get("/admin/customers/cust1/issues").status_code == 403
    assert (
        c.patch("/admin/customers/cust1/issues/b1", json={"status": "completed"}).status_code == 403
    )


def test_admin_note_only_leaves_status_alone(as_admin, backlog):
    r = TestClient(app).patch("/admin/customers/cust1/issues/b1", json={"note": "đang kiểm tra"})
    assert r.status_code == 200
    assert r.json()["note"] == "đang kiểm tra" and r.json()["status"] == "pending"
    assert backlog.calls == [{"note": "đang kiểm tra"}]


def test_admin_explicit_null_due_date_is_sent_as_clear(as_admin, backlog):
    c = TestClient(app)
    assert (
        c.patch("/admin/customers/cust1/issues/b1", json={"due_date": "2026-11-05"}).json()[
            "due_date"
        ]
        == "2026-11-05"
    )
    r = c.patch("/admin/customers/cust1/issues/b1", json={"due_date": None})
    assert r.status_code == 200 and r.json()["due_date"] is None
    assert backlog.calls[-1] == {"due_date": None}


def test_admin_empty_patch_is_422(as_admin, backlog):
    c = TestClient(app)
    assert c.patch("/admin/customers/cust1/issues/b1", json={}).status_code == 422
    # a null status means "leave it", so on its own it is nothing to do as well
    assert c.patch("/admin/customers/cust1/issues/b1", json={"status": None}).status_code == 422
    assert backlog.calls == []


def test_admin_bad_due_date_is_422(as_admin, backlog):
    r = TestClient(app).patch("/admin/customers/cust1/issues/b1", json={"due_date": "05/11/2026"})
    assert r.status_code == 422


def test_user_detail_shows_note_and_due_date(as_user, backlog):
    from datetime import date

    backlog.rows["b1"].note = "Đã sửa ở bản 2.3"
    backlog.rows["b1"].due_date = date(2026, 11, 5)
    body = TestClient(app).get("/widget/me/issues/b1").json()
    assert body["note"] == "Đã sửa ở bản 2.3" and body["due_date"] == "2026-11-05"

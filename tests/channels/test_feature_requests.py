import base64

import pytest
from fastapi.testclient import TestClient

from agent_customer_support.channels.deps import (
    get_attachment_store,
    get_customer_registry,
    get_mantis_client,
    get_request_backlog,
)
from agent_customer_support.mantis import MantisIssue
from agent_customer_support.models import (
    CustomerProfile,
    FileRef,
    RequestFile,
    RequestRecord,
)
from agent_customer_support.server import app

APP = "Lấy mẫu - Quan trắc"


class FakeBacklog:
    def __init__(self, *rows: RequestRecord):
        self.rows = {r.id: r for r in rows}
        self.calls: list[dict] = []

    async def add(self, *, id=None, files=None, **kw):
        rec = RequestRecord(id=id or f"f{len(self.rows) + 1}", files=files or [], **kw)
        self.rows[rec.id] = rec
        return rec

    async def get(self, request_id):
        r = self.rows.get(request_id)
        return r.model_copy(deep=True) if r else None

    async def list_by_type(self, customer_id, type):
        rows = [r for r in self.rows.values() if r.customer_id == customer_id and r.type == type]
        rows.sort(key=lambda r: r.created_at, reverse=True)
        return [r.model_dump(exclude={"transcript"}) for r in rows]

    async def update_issue(self, request_id, customer_id, *, by, type="bug", **changes):
        r = self.rows.get(request_id)
        if r is None or r.customer_id != customer_id or r.type != type:
            return None
        self.calls.append(changes)
        if "status" in changes:
            r.status = changes["status"]
            r.status_updated_by = by
        if "note" in changes:
            r.note = changes["note"]
        if "due_date" in changes:
            r.due_date = changes["due_date"]
        if "category" in changes:
            r.category = changes["category"]
        return r.model_copy(deep=True)


class FakeStore:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.puts: list[tuple] = []

    async def put_request_file(self, request_id, index, name, media_type, data):
        if self.fail:
            raise RuntimeError("s3 down")
        self.puts.append((request_id, index, name, media_type, data))
        key = f"requests/{request_id}/{index}-x"
        return RequestFile(name=name, media_type=media_type, s3_key=key, size_bytes=len(data))

    async def presign_request_file(self, f):
        return FileRef(
            name=f.name,
            media_type=f.media_type,
            size_bytes=f.size_bytes,
            url=f"https://s/{f.s3_key}",
        )


class FakeMantis:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list[dict] = []

    async def create_feature_request(self, **kw):
        self.calls.append(kw)
        return None if self.fail else MantisIssue(id=42, url="https://mantis/view.php?id=42")


class FakeRegistry:
    async def get(self, customer_id):
        return (
            CustomerProfile(customer_id=customer_id, name="X") if customer_id != "ghost" else None
        )


MINE = RequestRecord(
    id="r1",
    customer_id="cust1",
    type="new_feature",
    status="received",
    summary="thêm cột",
    title="Cột mới",
)
THEIRS = RequestRecord(id="r2", customer_id="other", type="new_feature", summary="của người khác")
BUG = RequestRecord(id="b1", customer_id="cust1", type="bug", summary="lỗi")


@pytest.fixture
def mantis():
    m = FakeMantis()
    app.dependency_overrides[get_mantis_client] = lambda: m
    yield m
    app.dependency_overrides.pop(get_mantis_client, None)


@pytest.fixture
def store():
    s = FakeStore()
    app.dependency_overrides[get_attachment_store] = lambda: s
    yield s
    app.dependency_overrides.pop(get_attachment_store, None)


@pytest.fixture
def backlog(store):
    b = FakeBacklog(MINE.model_copy(deep=True), THEIRS, BUG)
    app.dependency_overrides[get_request_backlog] = lambda: b
    app.dependency_overrides[get_customer_registry] = lambda: FakeRegistry()
    yield b
    for dep in (get_request_backlog, get_customer_registry):
        app.dependency_overrides.pop(dep, None)


@pytest.fixture
def customer(as_user):
    as_user.enabled_applications = [APP]
    as_user.mantis_handler_name = "manager1"
    return as_user


XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _file(name="mau.xlsx", media_type=XLSX, data=b"hello"):
    return {"name": name, "media_type": media_type, "data": base64.b64encode(data).decode()}


def _form(**kw):
    return {
        "title": "Xuất Excel",
        "requester_name": "Nguyễn Văn A",
        "requester_position": "Trưởng phòng",
        "requester_department": "Phòng Thí nghiệm",
        "application": APP,
        "summary": "Muốn xuất báo cáo ra Excel",
        "reason": "Để gửi cho sở",
    } | kw


# ---- customer: create ----


def test_create_files_ticket_and_stores_its_id(customer, backlog, mantis):
    r = TestClient(app).post("/widget/me/feature-requests", json=_form())
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "received" and body["mantis_issue_id"] == 42
    assert body["title"] == "Xuất Excel" and body["application"] == APP
    row = backlog.rows[body["id"]]
    assert row.type == "new_feature" and row.customer_id == "cust1"
    assert mantis.calls[0]["handler_name"] == "manager1"
    assert mantis.calls[0]["customer_id"] == "cust1"
    assert body["requester"] == {
        "name": "Nguyễn Văn A",
        "position": "Trưởng phòng",
        "department": "Phòng Thí nghiệm",
    }
    assert body["reason"] == "Để gửi cho sở"
    assert mantis.calls[0]["requester"].department == "Phòng Thí nghiệm"


def test_create_with_files_stores_them_and_sends_them_to_mantis(customer, backlog, mantis, store):
    form = _form(files=[_file(), _file("anh.png", "image/png", b"png")])
    r = TestClient(app).post("/widget/me/feature-requests", json=form)
    assert r.status_code == 201
    body = r.json()
    # stored under the row's own id, before the row was written
    assert [(p[0], p[2]) for p in store.puts] == [(body["id"], "mau.xlsx"), (body["id"], "anh.png")]
    assert store.puts[0][4] == b"hello"
    assert [f.name for f in mantis.calls[0]["files"]] == ["mau.xlsx", "anh.png"]
    assert [f["name"] for f in body["files"]] == ["mau.xlsx", "anh.png"]
    assert body["files"][0]["url"].startswith("https://s/requests/")
    assert len(backlog.rows[body["id"]].files) == 2


@pytest.mark.parametrize(
    "f",
    [
        _file("virus.exe", "application/octet-stream"),
        _file("bao_cao.pdf", "application/pdf"),
        # extension and media type must agree
        _file("anh.png", XLSX),
    ],
)
def test_create_rejects_file_types_outside_the_allow_list(customer, backlog, mantis, store, f):
    r = TestClient(app).post("/widget/me/feature-requests", json=_form(files=[f]))
    assert r.status_code == 422
    assert store.puts == [] and mantis.calls == []


def test_create_rejects_too_many_files(customer, backlog, mantis, store):
    r = TestClient(app).post("/widget/me/feature-requests", json=_form(files=[_file()] * 6))
    assert r.status_code == 422


def test_create_rejects_a_file_over_the_cap(customer, backlog, mantis, store, monkeypatch):
    from agent_customer_support.config import get_settings

    monkeypatch.setattr(get_settings(), "max_attachment_bytes", 4)
    r = TestClient(app).post("/widget/me/feature-requests", json=_form(files=[_file()]))
    assert r.status_code == 413
    assert store.puts == [] and mantis.calls == []


def test_failed_file_upload_saves_nothing(customer, backlog, mantis):
    s = FakeStore(fail=True)
    app.dependency_overrides[get_attachment_store] = lambda: s
    before = set(backlog.rows)
    r = TestClient(app).post("/widget/me/feature-requests", json=_form(files=[_file()]))
    assert r.status_code == 502
    assert set(backlog.rows) == before and mantis.calls == []


@pytest.mark.parametrize("field", ["requester_name", "requester_position", "requester_department"])
def test_create_requires_the_requester(customer, backlog, mantis, field):
    form = _form()
    del form[field]
    assert TestClient(app).post("/widget/me/feature-requests", json=form).status_code == 422
    assert (
        TestClient(app).post("/widget/me/feature-requests", json=_form(**{field: "  "})).status_code
        == 422
    )
    assert mantis.calls == []


def test_reason_is_optional(customer, backlog, mantis):
    form = _form()
    del form["reason"]
    r = TestClient(app).post("/widget/me/feature-requests", json=form)
    assert r.status_code == 201 and r.json()["reason"] is None


def test_create_still_saves_when_mantis_fails(customer, backlog, store):
    m = FakeMantis(fail=True)
    app.dependency_overrides[get_mantis_client] = lambda: m
    try:
        r = TestClient(app).post("/widget/me/feature-requests", json=_form())
    finally:
        app.dependency_overrides.pop(get_mantis_client, None)
    assert r.status_code == 201
    assert r.json()["mantis_issue_id"] is None
    assert r.json()["id"] in backlog.rows


def test_create_with_application_not_enabled_is_422(customer, backlog, mantis):
    r = TestClient(app).post("/widget/me/feature-requests", json=_form(application="Khác"))
    assert r.status_code == 422
    assert mantis.calls == []


@pytest.mark.parametrize("field", ["title", "summary"])
def test_create_with_blank_field_is_422(customer, backlog, mantis, field):
    r = TestClient(app).post("/widget/me/feature-requests", json=_form(**{field: "   "}))
    assert r.status_code == 422
    assert mantis.calls == []


def test_create_ignores_a_customer_id_in_the_body(customer, backlog, mantis):
    r = TestClient(app).post("/widget/me/feature-requests", json=_form(customer_id="other"))
    assert r.status_code == 201
    assert backlog.rows[r.json()["id"]].customer_id == "cust1"


# ---- customer: read ----


def test_user_lists_only_own_feature_requests(as_user, backlog):
    r = TestClient(app).get("/widget/me/feature-requests")
    assert r.status_code == 200
    assert [i["id"] for i in r.json()] == ["r1"]


@pytest.mark.parametrize("request_id", ["r2", "b1", "missing"])
def test_user_cannot_open_other_customers_or_bug_rows(as_user, backlog, request_id):
    assert TestClient(app).get(f"/widget/me/feature-requests/{request_id}").status_code == 404


def test_feature_row_is_not_listed_as_a_bug(as_user, backlog):
    assert TestClient(app).get("/widget/me/issues/r1").status_code == 404


# ---- admin ----


def test_admin_lists_and_updates(as_admin, backlog):
    c = TestClient(app)
    assert [i["id"] for i in c.get("/admin/customers/cust1/feature-requests").json()] == ["r1"]
    r = c.patch(
        "/admin/customers/cust1/feature-requests/r1",
        json={"status": "responded", "note": "Dự kiến quý 1"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "responded" and r.json()["note"] == "Dự kiến quý 1"
    assert backlog.rows["r1"].status_updated_by == "admin"


@pytest.mark.parametrize("status", ["no_issue", "completed", "pending"])
def test_admin_bug_status_is_rejected_for_a_feature(as_admin, backlog, status):
    r = TestClient(app).patch("/admin/customers/cust1/feature-requests/r1", json={"status": status})
    assert r.status_code == 422


def test_admin_sets_and_clears_category(as_admin, backlog):
    c = TestClient(app)
    r = c.patch("/admin/customers/cust1/feature-requests/r1", json={"category": "upgrade"})
    assert r.status_code == 200 and r.json()["category"] == "upgrade"
    r = c.patch("/admin/customers/cust1/feature-requests/r1", json={"category": None})
    assert r.status_code == 200 and r.json()["category"] is None
    assert backlog.calls[-1] == {"category": None}
    bad = c.patch("/admin/customers/cust1/feature-requests/r1", json={"category": "other"})
    assert bad.status_code == 422


def test_admin_cannot_patch_a_bug_through_the_feature_route(as_admin, backlog):
    r = TestClient(app).patch(
        "/admin/customers/cust1/feature-requests/b1", json={"status": "processing"}
    )
    assert r.status_code == 404


def test_admin_empty_patch_is_422(as_admin, backlog):
    r = TestClient(app).patch("/admin/customers/cust1/feature-requests/r1", json={})
    assert r.status_code == 422
    assert backlog.calls == []


def test_admin_unknown_customer_is_404(as_admin, backlog):
    assert TestClient(app).get("/admin/customers/ghost/feature-requests").status_code == 404


def test_non_admin_cannot_use_admin_routes(as_user, backlog):
    c = TestClient(app)
    assert c.get("/admin/customers/cust1/feature-requests").status_code == 403
    assert (
        c.patch(
            "/admin/customers/cust1/feature-requests/r1", json={"status": "processing"}
        ).status_code
        == 403
    )


def test_an_invalid_row_is_skipped_not_a_500(as_user, backlog):
    async def broken(customer_id, type):
        good = MINE.model_dump(mode="json")
        return [good, good | {"id": "bad", "status": "nonsense"}]

    backlog.list_by_type = broken
    r = TestClient(app).get("/widget/me/feature-requests")
    assert r.status_code == 200 and [i["id"] for i in r.json()] == ["r1"]

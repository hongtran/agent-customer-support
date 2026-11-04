from datetime import UTC, datetime
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from openpyxl import load_workbook

from agent_customer_support.channels.deps import get_customer_registry, get_request_backlog
from agent_customer_support.models import CustomerProfile, RequestRecord
from agent_customer_support.server import app


class FakeBacklog:
    def __init__(self, *rows: RequestRecord):
        self.rows = {r.id: r for r in rows}
        self.ranges: list[tuple] = []

    async def list_by_type(self, customer_id, type, *, created_from=None, created_before=None):
        self.ranges.append((created_from, created_before))
        rows = [
            r
            for r in self.rows.values()
            if r.customer_id == customer_id
            and r.type == type
            and (created_from is None or r.created_at >= created_from)
            and (created_before is None or r.created_at < created_before)
        ]
        rows.sort(key=lambda r: r.created_at, reverse=True)
        return [{"id": r.id} for r in rows]

    async def get_many(self, ids):
        return [self.rows[i] for i in ids if i in self.rows]


class FakeRegistry:
    async def get(self, customer_id):
        if customer_id == "ghost":
            return None
        return CustomerProfile(customer_id=customer_id, name="Công ty Một")


def _at(day: int, hour: int = 3) -> datetime:
    return datetime(2026, 11, day, hour, tzinfo=UTC)


ROWS = [
    RequestRecord(
        id="b1", customer_id="cust1", type="bug", summary="s", title="Bug 3/11", created_at=_at(3)
    ),
    # 17:30 UTC on the 4th is 00:30 on the 5th in Vietnam: outside a range ending the 4th
    RequestRecord(
        id="b2",
        customer_id="cust1",
        type="bug",
        summary="s",
        title="Bug 5/11 VN",
        created_at=_at(4, 17),
    ),
    RequestRecord(
        id="b3", customer_id="other", type="bug", summary="s", title="Other", created_at=_at(3)
    ),
    RequestRecord(
        id="f1",
        customer_id="cust1",
        type="new_feature",
        summary="s",
        title="Feature",
        created_at=_at(3),
    ),
]


@pytest.fixture
def backlog():
    b = FakeBacklog(*ROWS)
    app.dependency_overrides[get_request_backlog] = lambda: b
    app.dependency_overrides[get_customer_registry] = lambda: FakeRegistry()
    yield b
    for dep in (get_request_backlog, get_customer_registry):
        app.dependency_overrides.pop(dep, None)


def _titles(content: bytes) -> list[str]:
    ws = load_workbook(BytesIO(content)).active
    header = [c.value for c in ws[3]]
    col = header.index("Tiêu đề")
    return [row[col].value for row in ws.iter_rows(min_row=4)]


def test_issue_export_is_an_xlsx_of_this_customers_bugs_in_range(as_admin, backlog):
    r = TestClient(app).get(
        "/admin/customers/cust1/issues/export",
        params={"date_from": "2026-11-01", "date_to": "2026-11-04"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    assert (
        r.headers["content-disposition"]
        == "attachment; filename*=UTF-8''su-co_cust1_2026-11-01_2026-11-04.xlsx"
    )
    assert _titles(r.content) == ["Bug 3/11"]
    # the range is Vietnamese days: 01/11 00:00 to 05/11 00:00 at +07:00
    lo, hi = backlog.ranges[-1]
    assert lo == datetime(2026, 10, 31, 17, tzinfo=UTC)
    assert hi == datetime(2026, 11, 4, 17, tzinfo=UTC)


def test_export_without_dates_takes_everything(as_admin, backlog):
    r = TestClient(app).get("/admin/customers/cust1/issues/export")
    assert r.status_code == 200
    assert _titles(r.content) == ["Bug 5/11 VN", "Bug 3/11"]
    assert backlog.ranges[-1] == (None, None)
    assert "filename*=UTF-8''su-co_cust1.xlsx" in r.headers["content-disposition"]


def test_feature_export_has_only_feature_rows(as_admin, backlog):
    r = TestClient(app).get("/admin/customers/cust1/feature-requests/export")
    assert r.status_code == 200
    assert _titles(r.content) == ["Feature"]
    assert "tinh-nang_cust1" in r.headers["content-disposition"]


@pytest.mark.parametrize("path", ["issues", "feature-requests"])
def test_from_after_to_is_422(as_admin, backlog, path):
    r = TestClient(app).get(
        f"/admin/customers/cust1/{path}/export",
        params={"date_from": "2026-11-05", "date_to": "2026-11-01"},
    )
    assert r.status_code == 422


@pytest.mark.parametrize("path", ["issues", "feature-requests"])
def test_unknown_customer_is_404(as_admin, backlog, path):
    assert TestClient(app).get(f"/admin/customers/ghost/{path}/export").status_code == 404


@pytest.mark.parametrize("path", ["issues", "feature-requests"])
def test_non_admin_cannot_export(as_user, backlog, path):
    assert TestClient(app).get(f"/admin/customers/cust1/{path}/export").status_code == 403


def test_bad_date_is_422(as_admin, backlog):
    r = TestClient(app).get(
        "/admin/customers/cust1/issues/export", params={"date_from": "04/11/2026"}
    )
    assert r.status_code == 422

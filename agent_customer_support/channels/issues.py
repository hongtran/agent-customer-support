"""Verified-bug issues, for the customer who reported them and for CS.

An issue is a `RequestRecord` with `type == "bug"`, written by
`Coordinator._step_file_ticket`. Both lists are per-customer queries on the backlog's
by-customer index; there is deliberately no all-customers list, which would have to scan
every tenant's rows (transcripts included) on each load.
"""

from collections.abc import Callable
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field

from agent_customer_support import exports, image_urls
from agent_customer_support.channels.deps import (
    get_current_customer,
    get_customer_registry,
    get_doc_image_store,
    get_request_backlog,
    require_admin,
)
from agent_customer_support.config import get_settings
from agent_customer_support.models import (
    CustomerProfile,
    IssueDetail,
    IssueStatus,
    IssueSummary,
    RequestRecord,
    RequestType,
)
from agent_customer_support.stores.customer_registry import CustomerRegistry
from agent_customer_support.stores.doc_image_store import DocImageStore
from agent_customer_support.stores.request_backlog import RequestBacklog

widget_router = APIRouter(prefix="/widget/me/issues", tags=["widget"])

admin_router = APIRouter(
    prefix="/admin/customers/{customer_id}/issues",
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)


class IssuePatch(BaseModel):
    """An admin's edit. A field left out is not touched; an explicit null clears it
    (`model_fields_set` tells the two apart). A null status is the same as leaving it out."""

    status: IssueStatus | None = None
    note: str | None = Field(default=None, max_length=2000)
    due_date: date | None = None


async def load_request(
    backlog: RequestBacklog, request_id: str, customer_id: str, type: RequestType
) -> RequestRecord:
    """The row, or 404 when it is missing, of another type, or another customer's. One
    answer for all three, so an id outside the caller's tenant is not confirmed to exist.
    Shared with `channels/feature_requests.py`."""
    rec = await backlog.get(request_id)
    if rec is None or rec.type != type or rec.customer_id != customer_id:
        raise HTTPException(status_code=404, detail="not found")
    return rec


def patch_changes(body: BaseModel) -> dict[str, Any]:
    """The fields an admin PATCH actually sent. A field left out is not in the dict; an
    explicit null is (it clears). A null status means "leave it" and is dropped. Nothing
    left is a 422. Shared with `channels/feature_requests.py`."""
    changes: dict[str, Any] = {f: getattr(body, f) for f in body.model_fields_set}
    if "status" in changes and changes["status"] is None:
        del changes["status"]
    if not changes:
        raise HTTPException(status_code=422, detail="nothing to update")
    return changes


XLSX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


async def export_response(
    *,
    customer_id: str,
    type: RequestType,
    date_from: date | None,
    date_to: date | None,
    registry: CustomerRegistry,
    backlog: RequestBacklog,
    build: Callable[..., bytes],
    file_prefix: str,
) -> Response:
    """One customer's rows of one type, created between two days (both inclusive, as
    days in `usage_timezone`), as an .xlsx download. Shared with
    `channels/feature_requests.py`.

    The index gives the ids in the range; `get_many` then reads the full rows, because
    the index does not project note, due date, requester… Still one customer only, so
    no table scan.
    """
    if date_from and date_to and date_from > date_to:
        raise HTTPException(status_code=422, detail="date_from is after date_to")
    customer = await registry.get(customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="customer not found")

    tz = ZoneInfo(get_settings().usage_timezone)
    lo = datetime.combine(date_from, time(), tz) if date_from else None
    hi = datetime.combine(date_to + timedelta(days=1), time(), tz) if date_to else None
    items = await backlog.list_by_type(customer_id, type, created_from=lo, created_before=hi)
    records = await backlog.get_many([i["id"] for i in items])

    data = build(
        records,
        customer_name=customer.name or customer_id,
        date_from=date_from,
        date_to=date_to,
        tz=tz,
        now=datetime.now(UTC),
    )
    parts = [file_prefix, customer_id, *(d.isoformat() for d in (date_from, date_to) if d)]
    filename = "_".join(parts) + ".xlsx"
    return Response(
        content=data,
        media_type=XLSX_MEDIA_TYPE,
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"},
    )


async def _load_bug(backlog: RequestBacklog, issue_id: str, customer_id: str) -> RequestRecord:
    return await load_request(backlog, issue_id, customer_id, "bug")


async def _detail(rec: RequestRecord, doc_images: DocImageStore) -> IssueDetail:
    """The row with its stored transcript. Assistant lines keep `[[img:…]]` markers in
    storage; they are signed here so the reader sees pictures, not markers."""
    transcript = await image_urls.resolve_doc_images(doc_images, rec.transcript)
    return IssueDetail.model_validate(rec.model_dump() | {"transcript": transcript})


# ---- customer ------------------------------------------------------------------------


@widget_router.get("")
async def list_my_issues(
    customer: CustomerProfile = Depends(get_current_customer),
    backlog: RequestBacklog = Depends(get_request_backlog),
) -> list[IssueSummary]:
    """The caller's reported bugs, newest first. No path id: the tenant is the token's."""
    return await backlog.list_bugs(customer.customer_id)


@widget_router.get("/{issue_id}")
async def get_my_issue(
    issue_id: str,
    customer: CustomerProfile = Depends(get_current_customer),
    backlog: RequestBacklog = Depends(get_request_backlog),
    doc_images: DocImageStore = Depends(get_doc_image_store),
) -> IssueDetail:
    rec = await _load_bug(backlog, issue_id, customer.customer_id)
    return await _detail(rec, doc_images)


# ---- admin ---------------------------------------------------------------------------


@admin_router.get("")
async def list_customer_issues(
    customer_id: str,
    registry: CustomerRegistry = Depends(get_customer_registry),
    backlog: RequestBacklog = Depends(get_request_backlog),
) -> list[IssueSummary]:
    """A customer's reported bugs, newest first."""
    if await registry.get(customer_id) is None:
        raise HTTPException(status_code=404, detail="customer not found")
    return await backlog.list_bugs(customer_id)


# Registered before "/{issue_id}", or "export" would be read as an issue id.
@admin_router.get("/export")
async def export_customer_issues(
    customer_id: str,
    date_from: date | None = None,
    date_to: date | None = None,
    registry: CustomerRegistry = Depends(get_customer_registry),
    backlog: RequestBacklog = Depends(get_request_backlog),
) -> Response:
    """A customer's bugs created in a date range, as an Excel file."""
    return await export_response(
        customer_id=customer_id,
        type="bug",
        date_from=date_from,
        date_to=date_to,
        registry=registry,
        backlog=backlog,
        build=exports.issues_xlsx,
        file_prefix="su-co",
    )


@admin_router.get("/{issue_id}")
async def get_customer_issue(
    customer_id: str,
    issue_id: str,
    backlog: RequestBacklog = Depends(get_request_backlog),
    doc_images: DocImageStore = Depends(get_doc_image_store),
) -> IssueDetail:
    rec = await _load_bug(backlog, issue_id, customer_id)
    return await _detail(rec, doc_images)


@admin_router.patch("/{issue_id}")
async def update_issue(
    customer_id: str,
    issue_id: str,
    body: IssuePatch,
    admin: CustomerProfile = Depends(require_admin),
    backlog: RequestBacklog = Depends(get_request_backlog),
    doc_images: DocImageStore = Depends(get_doc_image_store),
) -> IssueDetail:
    """Set the status, note or due date the customer sees. A status change is recorded
    in the row's history. Nothing else happens: no mail, no MantisBT note."""
    changes = patch_changes(body)
    rec = await backlog.update_issue(issue_id, customer_id, by=admin.customer_id, **changes)
    if rec is None:
        raise HTTPException(status_code=404, detail="issue not found")
    return await _detail(rec, doc_images)

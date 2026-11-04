"""Feature requests: a customer asks for a new feature from a form, and CS answers with
a status, a note and a due date.

A feature request is a `RequestRecord` with `type == "new_feature"`. Unlike a bug it is
written here, not by the agent: no chat, no LLM call. On create the MantisBT ticket is
filed first, so the row is written once with the ticket id (same order as
`Coordinator._step_file_ticket`); a tracker outage costs the ticket id, not the request.

Files (Excel, Word, images) are checked here, stored in S3 under `requests/<id>/`
before the ticket, and sent to MantisBT as a note. A failed S3 upload fails the whole
create (502) and saves nothing, since the customer can simply retry; a failed MantisBT
upload costs only the files on the ticket.

Lists are per-customer queries on the backlog's by-customer index, like the issue lists
in `channels/issues.py`; there is no all-customers list on purpose.
"""

import base64
import logging
import uuid
from binascii import Error as BinasciiError
from datetime import date
from pathlib import PurePath
from typing import Final, Literal

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field, ValidationError, ValidationInfo, field_validator

from agent_customer_support.channels.deps import (
    get_attachment_store,
    get_current_customer,
    get_customer_registry,
    get_mantis_client,
    get_request_backlog,
    require_admin,
)
from agent_customer_support import exports
from agent_customer_support.channels.issues import export_response, load_request, patch_changes
from agent_customer_support.config import get_settings
from agent_customer_support.mantis import MantisClient, MantisFile
from agent_customer_support.models import (
    CustomerProfile,
    FeatureCategory,
    FeatureDetail,
    FeatureStatus,
    FeatureSummary,
    RequestFile,
    RequestRecord,
    Requester,
)
from agent_customer_support.stores.attachment_store import AttachmentStore
from agent_customer_support.stores.customer_registry import CustomerRegistry
from agent_customer_support.stores.request_backlog import RequestBacklog

widget_router = APIRouter(prefix="/widget/me/feature-requests", tags=["widget"])

admin_router = APIRouter(
    prefix="/admin/customers/{customer_id}/feature-requests",
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)

logger = logging.getLogger(__name__)

_TYPE: Final[Literal["new_feature"]] = "new_feature"

MAX_FILES = 5

# Extension -> the media types a browser may report for it. Both must match: the
# extension is what the user sees, the media type is what we store and serve.
_ALLOWED_FILES: dict[str, set[str]] = {
    ".xlsx": {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"},
    ".xls": {"application/vnd.ms-excel"},
    ".docx": {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
    ".doc": {"application/msword"},
    ".png": {"image/png"},
    ".jpg": {"image/jpeg"},
    ".jpeg": {"image/jpeg"},
}


class UploadFile(BaseModel):
    """One file from the form, as base64 in the JSON body (same as chat attachments)."""

    name: str = Field(min_length=1, max_length=200)
    media_type: str
    data: str  # base64

    @field_validator("data")
    @classmethod
    def _valid_base64(cls, v: str) -> str:
        try:
            base64.b64decode(v, validate=True)
        except (BinasciiError, ValueError) as exc:
            raise ValueError(f"data is not valid base64: {exc}") from exc
        return v

    @field_validator("media_type")
    @classmethod
    def _allowed(cls, v: str, info: ValidationInfo) -> str:
        ext = PurePath(info.data.get("name", "")).suffix.lower()
        if v not in _ALLOWED_FILES.get(ext, set()):
            raise ValueError("only Excel, Word and image files (png, jpg) are allowed")
        return v

    @property
    def decoded_size(self) -> int:
        """Decoded byte count from the base64 length, without decoding."""
        return len(self.data.rstrip("=")) * 3 // 4


class FeatureRequestCreate(BaseModel):
    """The customer's form. No customer id: the tenant is the token's. The requester is
    a person in the customer's company: the login is shared, so CS needs the name."""

    title: str = Field(min_length=1, max_length=200)
    requester_name: str = Field(min_length=1, max_length=100)
    requester_position: str = Field(min_length=1, max_length=100)
    requester_department: str = Field(min_length=1, max_length=100)
    # A display name, one of the customer's `enabled_applications`.
    application: str = Field(min_length=1)
    # "Nội dung yêu cầu".
    summary: str = Field(min_length=1, max_length=4000)
    reason: str | None = Field(default=None, max_length=2000)
    files: list[UploadFile] = Field(default_factory=list, max_length=MAX_FILES)


class FeaturePatch(BaseModel):
    """An admin's edit; same rules as `IssuePatch` (left out = untouched, null = clear).
    `note` is shown as "Giải pháp"."""

    status: FeatureStatus | None = None
    note: str | None = Field(default=None, max_length=2000)
    due_date: date | None = None
    category: FeatureCategory | None = None


def _summaries(items: list[dict]) -> list[FeatureSummary]:
    """List rows, skipping (and logging) any row that does not validate, so one bad
    row hides itself instead of failing the whole page with a 500."""
    out = []
    for i in items:
        try:
            out.append(FeatureSummary.model_validate(i))
        except ValidationError as exc:
            logger.warning("skipping invalid feature request row %s: %s", i.get("id"), exc)
    return out


async def _detail(rec: RequestRecord, store: AttachmentStore) -> FeatureDetail:
    """The row with its files signed for download. A file that fails to sign is left
    out rather than failing the page."""
    refs = []
    for f in rec.files:
        try:
            refs.append(await store.presign_request_file(f))
        except Exception as exc:  # noqa: BLE001 - one bad file must not hide the request
            logger.warning("presign failed for %s: %s", f.s3_key, exc)
    data = rec.model_dump(exclude={"files"}) | {"files": refs}
    return FeatureDetail.model_validate(data)


# ---- customer ------------------------------------------------------------------------


@widget_router.post("", status_code=201)
async def create_feature_request(
    body: FeatureRequestCreate,
    customer: CustomerProfile = Depends(get_current_customer),
    backlog: RequestBacklog = Depends(get_request_backlog),
    mantis: MantisClient = Depends(get_mantis_client),
    store: AttachmentStore = Depends(get_attachment_store),
) -> FeatureDetail:
    title = body.title.strip()
    summary = body.summary.strip()
    requester = Requester(
        name=body.requester_name.strip(),
        position=body.requester_position.strip(),
        department=body.requester_department.strip(),
    )
    if not (title and summary and requester.name and requester.position and requester.department):
        raise HTTPException(status_code=422, detail="required field is blank")
    # Only a module the customer bought: the request is filed under it, and asking for
    # a feature in someone else's product makes no sense.
    if body.application not in customer.enabled_applications:
        raise HTTPException(status_code=422, detail="application not enabled for this customer")
    cap = get_settings().max_attachment_bytes
    if any(f.decoded_size > cap for f in body.files):
        raise HTTPException(status_code=413, detail=f"file larger than {cap} bytes")

    # The id comes first: the files are stored under it before the row exists.
    request_id = str(uuid.uuid4())
    stored: list[RequestFile] = []
    try:
        for i, f in enumerate(body.files):
            raw = base64.b64decode(f.data)
            stored.append(await store.put_request_file(request_id, i, f.name, f.media_type, raw))
    except Exception as exc:  # noqa: BLE001 - nothing is saved yet; the user can retry
        logger.warning("feature request file upload failed for %s: %s", customer.customer_id, exc)
        raise HTTPException(status_code=502, detail="file upload failed") from exc

    reason = (body.reason or "").strip() or None
    issue = await mantis.create_feature_request(
        title=title,
        summary=summary,
        requester=requester,
        customer_id=customer.customer_id,
        customer_name=customer.name,
        application=body.application,
        reason=reason,
        files=[MantisFile(name=f.name, content_b64=f.data) for f in body.files],
        handler_name=customer.mantis_handler_name,
    )
    rec = await backlog.add(
        id=request_id,
        customer_id=customer.customer_id,
        type=_TYPE,
        status="received",
        title=title,
        summary=summary,
        application=body.application,
        requester=requester,
        reason=reason,
        files=stored,
        mantis_issue_id=issue.id if issue else None,
        mantis_issue_url=issue.url if issue else None,
    )
    return await _detail(rec, store)


@widget_router.get("")
async def list_my_feature_requests(
    customer: CustomerProfile = Depends(get_current_customer),
    backlog: RequestBacklog = Depends(get_request_backlog),
) -> list[FeatureSummary]:
    """The caller's feature requests, newest first."""
    items = await backlog.list_by_type(customer.customer_id, _TYPE)
    return _summaries(items)


@widget_router.get("/{request_id}")
async def get_my_feature_request(
    request_id: str,
    customer: CustomerProfile = Depends(get_current_customer),
    backlog: RequestBacklog = Depends(get_request_backlog),
    store: AttachmentStore = Depends(get_attachment_store),
) -> FeatureDetail:
    rec = await load_request(backlog, request_id, customer.customer_id, _TYPE)
    return await _detail(rec, store)


# ---- admin ---------------------------------------------------------------------------


@admin_router.get("")
async def list_customer_feature_requests(
    customer_id: str,
    registry: CustomerRegistry = Depends(get_customer_registry),
    backlog: RequestBacklog = Depends(get_request_backlog),
) -> list[FeatureSummary]:
    """One customer's feature requests, newest first."""
    if await registry.get(customer_id) is None:
        raise HTTPException(status_code=404, detail="customer not found")
    items = await backlog.list_by_type(customer_id, _TYPE)
    return _summaries(items)


# Registered before "/{request_id}", or "export" would be read as a request id.
@admin_router.get("/export")
async def export_customer_feature_requests(
    customer_id: str,
    date_from: date | None = None,
    date_to: date | None = None,
    registry: CustomerRegistry = Depends(get_customer_registry),
    backlog: RequestBacklog = Depends(get_request_backlog),
) -> Response:
    """A customer's feature requests created in a date range, as an Excel file."""
    return await export_response(
        customer_id=customer_id,
        type=_TYPE,
        date_from=date_from,
        date_to=date_to,
        registry=registry,
        backlog=backlog,
        build=exports.features_xlsx,
        file_prefix="tinh-nang",
    )


@admin_router.get("/{request_id}")
async def get_customer_feature_request(
    customer_id: str,
    request_id: str,
    backlog: RequestBacklog = Depends(get_request_backlog),
    store: AttachmentStore = Depends(get_attachment_store),
) -> FeatureDetail:
    return await _detail(await load_request(backlog, request_id, customer_id, _TYPE), store)


@admin_router.patch("/{request_id}")
async def update_feature_request(
    customer_id: str,
    request_id: str,
    body: FeaturePatch,
    admin: CustomerProfile = Depends(require_admin),
    backlog: RequestBacklog = Depends(get_request_backlog),
    store: AttachmentStore = Depends(get_attachment_store),
) -> FeatureDetail:
    """Set the status, solution (`note`), due date or category the customer sees. A status change is recorded
    in the row's history. Nothing else happens: no mail, no MantisBT note."""
    changes = patch_changes(body)
    rec = await backlog.update_issue(
        request_id, customer_id, by=admin.customer_id, type=_TYPE, **changes
    )
    if rec is None:
        raise HTTPException(status_code=404, detail="not found")
    return await _detail(rec, store)

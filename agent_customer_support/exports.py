"""Excel exports of one customer's bugs ("Sự cố") and feature requests ("Tính năng mới").

Pure: rows in, `.xlsx` bytes out, no I/O. The routes in `channels/issues.py` and
`channels/feature_requests.py` read the rows and send the bytes.

Every text cell is written as a string, never as a formula. openpyxl stores any string
that starts with `=` as a formula, and the text here is written by customers (title,
content, reason…) — a value like `=HYPERLINK(...)` would otherwise run in CS's Excel.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from io import BytesIO
from typing import Any
from zoneinfo import ZoneInfo

from openpyxl import Workbook
from openpyxl.cell import Cell
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from agent_customer_support.models import RequestRecord

# Vietnamese labels. Copies of the maps in ui/components/IssueStatusBadge.tsx — keep
# them in sync.
STATUS_LABEL: dict[str, str] = {
    # bugs
    "pending": "Chờ xử lý",
    "completed": "Đã xử lý",
    "no_issue": "Không phải lỗi",
    # features
    "received": "Đã tiếp nhận",
    "responded": "Đã phản hồi",
    "updated": "Đã cập nhật",
    "awaiting_agreement": "Chờ thống nhất",
    # both
    "processing": "Đang xử lý",
}
CATEGORY_LABEL: dict[str, str] = {
    "modify": "Sửa đổi",
    "upgrade": "Nâng cấp",
    "extend_other_app": "Mở rộng ứng dụng khác",
}

_DATE_FMT = "dd/mm/yyyy"
_DATETIME_FMT = "dd/mm/yyyy hh:mm"
_HEADER_ROW = 3


@dataclass(frozen=True)
class _Col:
    header: str
    width: int
    value: Callable[[RequestRecord], Any]
    wrap: bool = False


def _contact(r: RequestRecord) -> str | None:
    c = r.contact
    if c is None:
        return None
    return " · ".join(v for v in (c.phone, c.email) if v) or c.raw or None


def _requester(field: str) -> Callable[[RequestRecord], str | None]:
    return lambda r: getattr(r.requester, field) if r.requester else None


_ISSUE_COLS = [
    _Col("Ngày tạo", 17, lambda r: r.created_at),
    _Col("Tiêu đề", 40, lambda r: r.title, wrap=True),
    _Col("Ứng dụng", 22, lambda r: r.application),
    _Col("Mô tả", 60, lambda r: r.summary, wrap=True),
    _Col("Trạng thái", 16, lambda r: STATUS_LABEL.get(r.status, r.status)),
    _Col("Ghi chú", 40, lambda r: r.note, wrap=True),
    _Col("Ngày dự kiến", 14, lambda r: r.due_date),
    _Col("Ngày hoàn thành", 17, lambda r: r.resolved_at),
    _Col("Mantis #", 10, lambda r: r.mantis_issue_id),
    _Col("Link Mantis", 40, lambda r: r.mantis_issue_url),
    _Col("Liên hệ", 30, _contact, wrap=True),
]

_FEATURE_COLS = [
    _Col("Ngày tạo", 17, lambda r: r.created_at),
    _Col("Tiêu đề", 40, lambda r: r.title, wrap=True),
    _Col("Tên nhân sự", 22, _requester("name")),
    _Col("Chức vụ", 18, _requester("position")),
    _Col("Phòng ban", 20, _requester("department")),
    _Col("Ứng dụng", 22, lambda r: r.application),
    _Col("Nội dung yêu cầu", 60, lambda r: r.summary, wrap=True),
    _Col("Lý do", 40, lambda r: r.reason, wrap=True),
    _Col("Số file", 8, lambda r: len(r.files)),
    _Col("Thuộc hạng mục", 22, lambda r: CATEGORY_LABEL.get(r.category or "", r.category)),
    _Col("Trạng thái", 16, lambda r: STATUS_LABEL.get(r.status, r.status)),
    _Col("Giải pháp", 40, lambda r: r.note, wrap=True),
    _Col("Ngày dự kiến hoàn thành", 14, lambda r: r.due_date),
    _Col("Ngày hoàn thành", 17, lambda r: r.resolved_at),
    _Col("Mantis #", 10, lambda r: r.mantis_issue_id),
    _Col("Link Mantis", 40, lambda r: r.mantis_issue_url),
]


def _set(cell: Cell, value: Any, tz: ZoneInfo) -> None:
    """Write one value with the right Excel type. A datetime is shown in `tz` and
    stored without tzinfo, which openpyxl refuses."""
    if isinstance(value, datetime):
        cell.value = value.astimezone(tz).replace(tzinfo=None)
        cell.number_format = _DATETIME_FMT
    elif isinstance(value, date):
        cell.value = value
        cell.number_format = _DATE_FMT
    elif isinstance(value, str):
        cell.value = value
        cell.data_type = "s"  # never a formula, see the module docstring
    else:
        cell.value = value


def _period(date_from: date | None, date_to: date | None) -> str:
    def fmt(d: date | None) -> str:
        return d.strftime("%d/%m/%Y") if d else "…"

    if not date_from and not date_to:
        return "tất cả"
    return f"{fmt(date_from)} – {fmt(date_to)}"


def _build(
    sheet_title: str,
    cols: list[_Col],
    records: list[RequestRecord],
    *,
    customer_name: str,
    date_from: date | None,
    date_to: date | None,
    tz: ZoneInfo,
    now: datetime,
) -> bytes:
    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = sheet_title

    title = ws.cell(row=1, column=1)
    _set(title, f"{sheet_title} — {customer_name}", tz)
    title.font = Font(bold=True, size=13)
    info = ws.cell(row=2, column=1)
    _set(
        info,
        f"Ngày tạo: {_period(date_from, date_to)} · {len(records)} dòng · "
        f"xuất lúc {now.astimezone(tz).strftime('%d/%m/%Y %H:%M')}",
        tz,
    )
    info.font = Font(italic=True, color="666666")

    headers = ["STT", *(c.header for c in cols)]
    widths = [6, *(c.width for c in cols)]
    fill = PatternFill("solid", fgColor="DDEBF7")
    for i, (h, w) in enumerate(zip(headers, widths, strict=True), start=1):
        cell = ws.cell(row=_HEADER_ROW, column=i, value=h)
        cell.font = Font(bold=True)
        cell.fill = fill
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        ws.column_dimensions[get_column_letter(i)].width = w

    top = Alignment(vertical="top")
    top_wrap = Alignment(vertical="top", wrap_text=True)
    for n, rec in enumerate(records, start=1):
        row = _HEADER_ROW + n
        ws.cell(row=row, column=1, value=n).alignment = top
        for i, col in enumerate(cols, start=2):
            cell = ws.cell(row=row, column=i)
            _set(cell, col.value(rec), tz)
            cell.alignment = top_wrap if col.wrap else top

    ws.freeze_panes = ws.cell(row=_HEADER_ROW + 1, column=1)
    last = get_column_letter(len(headers))
    ws.auto_filter.ref = f"A{_HEADER_ROW}:{last}{_HEADER_ROW + max(len(records), 1)}"

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


def issues_xlsx(
    records: list[RequestRecord],
    *,
    customer_name: str,
    date_from: date | None,
    date_to: date | None,
    tz: ZoneInfo,
    now: datetime,
) -> bytes:
    """One customer's bugs as an .xlsx file, one row each, in the order given."""
    return _build(
        "Sự cố",
        _ISSUE_COLS,
        records,
        customer_name=customer_name,
        date_from=date_from,
        date_to=date_to,
        tz=tz,
        now=now,
    )


def features_xlsx(
    records: list[RequestRecord],
    *,
    customer_name: str,
    date_from: date | None,
    date_to: date | None,
    tz: ZoneInfo,
    now: datetime,
) -> bytes:
    """One customer's feature requests as an .xlsx file, one row each, in the order given."""
    return _build(
        "Tính năng mới",
        _FEATURE_COLS,
        records,
        customer_name=customer_name,
        date_from=date_from,
        date_to=date_to,
        tz=tz,
        now=now,
    )

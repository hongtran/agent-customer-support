from datetime import UTC, date, datetime
from io import BytesIO
from zoneinfo import ZoneInfo

from openpyxl import load_workbook

from agent_customer_support import exports
from agent_customer_support.models import ContactInfo, RequestFile, RequestRecord, Requester

TZ = ZoneInfo("Asia/Ho_Chi_Minh")
NOW = datetime(2026, 11, 4, 3, 0, tzinfo=UTC)


def _sheet(data: bytes):
    return load_workbook(BytesIO(data)).active


def test_issue_sheet_has_header_labels_and_local_times():
    rec = RequestRecord(
        id="b1",
        customer_id="c1",
        type="bug",
        summary="Import lỗi 500",
        title="Lỗi import",
        status="completed",
        note="Đã sửa",
        due_date=date(2026, 11, 10),
        # 16:30 UTC is 23:30 in Vietnam, same day
        created_at=datetime(2026, 11, 3, 16, 30, tzinfo=UTC),
        mantis_issue_id=42,
        contact=ContactInfo(phone="0912345678", email="a@b.vn", raw="x"),
    )
    ws = _sheet(
        exports.issues_xlsx(
            [rec],
            customer_name="Công ty A",
            date_from=date(2026, 11, 1),
            date_to=date(2026, 11, 4),
            tz=TZ,
            now=NOW,
        )
    )
    assert ws["A1"].value == "Sự cố — Công ty A"
    assert "01/11/2026 – 04/11/2026" in ws["A2"].value
    header = [c.value for c in ws[3]]
    assert header[:3] == ["STT", "Ngày tạo", "Tiêu đề"]
    row = {h: c.value for h, c in zip(header, ws[4], strict=True)}
    assert row["STT"] == 1
    assert row["Ngày tạo"] == datetime(2026, 11, 3, 23, 30)
    assert row["Trạng thái"] == "Đã xử lý"
    assert row["Ghi chú"] == "Đã sửa"
    assert row["Ngày dự kiến"].date() == date(2026, 11, 10)
    assert row["Mantis #"] == 42
    assert row["Liên hệ"] == "0912345678 · a@b.vn"


def test_feature_sheet_has_requester_category_and_file_count():
    rec = RequestRecord(
        id="f1",
        customer_id="c1",
        type="new_feature",
        summary="Thêm cột",
        title="Cột mới",
        status="awaiting_agreement",
        category="extend_other_app",
        requester=Requester(name="Nguyễn Văn A", position="Trưởng phòng", department="QA"),
        reason="Cần báo cáo",
        files=[RequestFile(name="a.xlsx", media_type="x", s3_key="k", size_bytes=1)],
    )
    ws = _sheet(
        exports.features_xlsx(
            [rec], customer_name="A", date_from=None, date_to=None, tz=TZ, now=NOW
        )
    )
    assert "tất cả" in ws["A2"].value
    header = [c.value for c in ws[3]]
    row = {h: c.value for h, c in zip(header, ws[4], strict=True)}
    assert row["Tên nhân sự"] == "Nguyễn Văn A"
    assert row["Phòng ban"] == "QA"
    assert row["Thuộc hạng mục"] == "Mở rộng ứng dụng khác"
    assert row["Trạng thái"] == "Chờ thống nhất"
    assert row["Số file"] == 1
    assert row["Lý do"] == "Cần báo cáo"


def test_customer_text_is_never_a_formula():
    rec = RequestRecord(
        id="f1", customer_id="c1", type="new_feature", summary="=1+1", title='=HYPERLINK("x")'
    )
    data = exports.features_xlsx(
        [rec], customer_name="=A", date_from=None, date_to=None, tz=TZ, now=NOW
    )
    ws = _sheet(data)
    cells = [c for c in ws[4] if isinstance(c.value, str) and c.value.startswith("=")]
    assert {c.value for c in cells} == {"=1+1", '=HYPERLINK("x")'}
    assert all(c.data_type == "s" for c in cells)
    assert ws["A1"].data_type == "s"

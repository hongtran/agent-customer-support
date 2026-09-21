import pytest
from agent_customer_support.stores.request_backlog import RequestBacklog

pytestmark = pytest.mark.asyncio


async def test_add_request():
    rb = RequestBacklog()
    await rb.init()
    rec = await rb.add(
        customer_id="c1",
        type="feature",
        summary="thêm cột",
        application="Kinh doanh",
        transcript="...",
    )
    assert rec.id and rec.type == "feature"
    got = await rb.get(rec.id)
    assert got and got.summary == "thêm cột" and got.application == "Kinh doanh"
    # a row written before tickets existed has no title and no ticket
    assert got.title is None and got.mantis_issue_id is None and got.mantis_issue_url is None


async def test_add_bug_with_ticket_round_trips():
    rb = RequestBacklog()
    await rb.init()
    rec = await rb.add(
        customer_id="c1",
        type="bug",
        summary="Import file mẫu thì báo lỗi 500.",
        title="Import ký hiệu mẫu báo lỗi 500",
        application="Lấy mẫu - Quan trắc",
        transcript="user: lỗi",
        mantis_issue_id=42,
        mantis_issue_url="https://mantis.example/view.php?id=42",
    )
    got = await rb.get(rec.id)
    assert got is not None
    assert got.title == "Import ký hiệu mẫu báo lỗi 500"
    assert got.mantis_issue_id == 42
    assert got.mantis_issue_url == "https://mantis.example/view.php?id=42"


async def test_add_accepts_how_to_missing_type():
    """knowledge.py already writes this type; the store's Literal must allow it."""
    rb = RequestBacklog()
    await rb.init()
    rec = await rb.add(customer_id="c1", type="how_to_missing", summary="chưa có hướng dẫn")
    assert rec.type == "how_to_missing"


async def test_set_contact_updates_only_that_field():
    from agent_customer_support.models import ContactInfo

    rb = RequestBacklog()
    await rb.init()
    rec = await rb.add(customer_id="c1", type="bug", summary="lỗi", title="T", mantis_issue_id=9)
    await rb.set_contact(
        rec.id, ContactInfo(phone="0912345678", email="a@b.vn", raw="0912345678 a@b.vn")
    )
    got = await rb.get(rec.id)
    assert got is not None
    assert got.contact is not None
    assert (got.contact.phone, got.contact.email) == ("0912345678", "a@b.vn")
    # the rest of the row is untouched
    assert got.title == "T" and got.mantis_issue_id == 9 and got.summary == "lỗi"

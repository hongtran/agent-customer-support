import pytest
from agent_customer_support.stores.request_backlog import RequestBacklog

pytestmark = pytest.mark.asyncio


async def test_add_request():
    rb = RequestBacklog()
    await rb.init()
    rec = await rb.add(
        customer_id="c1",
        type="new_feature",
        summary="thêm cột",
        application="Kinh doanh",
        transcript="...",
    )
    assert rec.id and rec.type == "new_feature"
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


# ---- issue tracking ----------------------------------------------------------------


def _cid() -> str:
    # DynamoDB Local keeps rows between runs, so each test owns a fresh customer.
    import uuid

    return f"c-{uuid.uuid4()}"


async def test_new_row_is_pending():
    rb = RequestBacklog()
    await rb.init()
    rec = await rb.add(customer_id=_cid(), type="bug", summary="lỗi")
    assert rec.status == "pending"
    got = await rb.get(rec.id)
    assert got is not None and got.status == "pending" and got.status_updated_at is None


async def test_list_bugs_is_one_customers_bugs_newest_first():
    rb = RequestBacklog()
    await rb.init()
    me, other = _cid(), _cid()
    first = await rb.add(customer_id=me, type="bug", summary="một", title="T1")
    second = await rb.add(customer_id=me, type="bug", summary="hai", title="T2")
    await rb.add(customer_id=me, type="how_to_missing", summary="không phải lỗi")
    await rb.add(customer_id=other, type="bug", summary="của khách khác")

    items = await rb.list_bugs(me)
    assert [i.id for i in items] == [second.id, first.id]
    assert items[0].title == "T2" and items[0].status == "pending"


async def test_row_written_before_status_existed_lists_as_pending():
    from agent_customer_support.stores.dynamo import get_resource

    rb = RequestBacklog()
    await rb.init()
    me = _cid()
    async with get_resource() as ddb:
        table = await ddb.Table(rb.table_name)
        await table.put_item(
            Item={
                "id": f"old-{me}",
                "customer_id": me,
                "type": "bug",
                "summary": "cũ",
                "transcript": "",
                "created_at": "2026-01-01T00:00:00Z",
            }
        )
    [item] = await rb.list_bugs(me)
    assert item.status == "pending"


async def test_status_change_updates_and_returns_the_row():
    rb = RequestBacklog()
    await rb.init()
    me = _cid()
    rec = await rb.add(customer_id=me, type="bug", summary="lỗi", title="T", mantis_issue_id=7)
    updated = await rb.update_issue(rec.id, me, by="admin", status="processing")
    assert updated is not None
    assert updated.status == "processing" and updated.status_updated_by == "admin"
    assert updated.status_updated_at is not None
    # the rest of the row is untouched
    assert updated.title == "T" and updated.mantis_issue_id == 7
    [item] = await rb.list_bugs(me)
    assert item.status == "processing"


async def test_update_refuses_missing_wrong_customer_and_non_bug():
    rb = RequestBacklog()
    await rb.init()
    me = _cid()
    bug = await rb.add(customer_id=me, type="bug", summary="lỗi")
    howto = await rb.add(customer_id=me, type="how_to_missing", summary="hỏi")

    assert await rb.update_issue("no-such-id", me, by="admin", status="completed") is None
    assert await rb.get("no-such-id") is None  # the failed update created no stub row
    assert await rb.update_issue(bug.id, _cid(), by="admin", status="completed") is None
    assert await rb.update_issue(howto.id, me, by="admin", status="completed") is None
    got = await rb.get(bug.id)
    assert got is not None and got.status == "pending"


async def test_init_adds_the_index_to_a_table_that_predates_it():
    import uuid

    from agent_customer_support.stores.dynamo import ensure_table, get_resource
    from agent_customer_support.stores.request_backlog import BY_CUSTOMER_INDEX

    rb = RequestBacklog()
    rb.table_name = f"requests-legacy-{uuid.uuid4()}"
    await ensure_table(rb.table_name, key="id")  # the old shape: no index
    await rb.init()
    async with get_resource() as ddb:
        desc = await ddb.meta.client.describe_table(TableName=rb.table_name)
        await ddb.meta.client.delete_table(TableName=rb.table_name)
    names = {i["IndexName"] for i in desc["Table"].get("GlobalSecondaryIndexes", [])}
    assert BY_CUSTOMER_INDEX in names


async def test_status_changes_are_appended_to_the_history_in_order():
    rb = RequestBacklog()
    await rb.init()
    me = _cid()
    rec = await rb.add(customer_id=me, type="bug", summary="lỗi")
    await rb.update_issue(rec.id, me, by="cs1", status="processing")
    got = await rb.update_issue(rec.id, me, by="cs2", status="completed", note="Đã sửa ở v2.3")
    assert got is not None
    assert [(h.status, h.by, h.note) for h in got.status_history] == [
        ("processing", "cs1", None),
        ("completed", "cs2", "Đã sửa ở v2.3"),
    ]
    assert got.status_history[0].at <= got.status_history[1].at
    assert got.note == "Đã sửa ở v2.3"


async def test_resolved_at_is_set_on_close_and_removed_on_reopen():
    rb = RequestBacklog()
    await rb.init()
    me = _cid()
    rec = await rb.add(customer_id=me, type="bug", summary="lỗi")
    closed = await rb.update_issue(rec.id, me, by="cs", status="no_issue")
    assert closed is not None and closed.resolved_at is not None
    reopened = await rb.update_issue(rec.id, me, by="cs", status="processing")
    assert reopened is not None and reopened.resolved_at is None


async def test_note_alone_adds_no_history_and_blank_note_clears_it():
    rb = RequestBacklog()
    await rb.init()
    me = _cid()
    rec = await rb.add(customer_id=me, type="bug", summary="lỗi")
    got = await rb.update_issue(rec.id, me, by="cs", note="  đang kiểm tra  ")
    assert got is not None and got.note == "đang kiểm tra"
    assert got.status_history == [] and got.status == "pending"
    cleared = await rb.update_issue(rec.id, me, by="cs", note="   ")
    assert cleared is not None and cleared.note is None


async def test_due_date_round_trips_and_clears():
    from datetime import date

    rb = RequestBacklog()
    await rb.init()
    me = _cid()
    rec = await rb.add(customer_id=me, type="bug", summary="lỗi")
    got = await rb.update_issue(rec.id, me, by="cs", due_date=date(2026, 11, 5))
    assert got is not None and got.due_date == date(2026, 11, 5)
    cleared = await rb.update_issue(rec.id, me, by="cs", due_date=None)
    assert cleared is not None and cleared.due_date is None


async def test_fields_not_sent_are_left_alone():
    from datetime import date

    rb = RequestBacklog()
    await rb.init()
    me = _cid()
    rec = await rb.add(customer_id=me, type="bug", summary="lỗi")
    await rb.update_issue(rec.id, me, by="cs", note="ghi chú", due_date=date(2026, 11, 5))
    got = await rb.update_issue(rec.id, me, by="cs", status="processing")
    assert got is not None
    assert got.note == "ghi chú" and got.due_date == date(2026, 11, 5)


async def test_update_with_nothing_to_change_is_refused():
    rb = RequestBacklog()
    with pytest.raises(ValueError):
        await rb.update_issue("x", "c", by="cs")


async def test_row_without_history_attribute_appends():
    from agent_customer_support.stores.dynamo import get_resource

    rb = RequestBacklog()
    await rb.init()
    me = _cid()
    async with get_resource() as ddb:
        table = await ddb.Table(rb.table_name)
        await table.put_item(
            Item={
                "id": f"old-{me}",
                "customer_id": me,
                "type": "bug",
                "summary": "cũ",
                "created_at": "2026-01-01T00:00:00Z",
            }
        )
    got = await rb.update_issue(f"old-{me}", me, by="cs", status="completed")
    assert got is not None and [h.status for h in got.status_history] == ["completed"]


# ---- feature requests ----


async def test_list_by_type_returns_only_that_type():
    rb = RequestBacklog()
    await rb.init()
    me = _cid()
    await rb.add(customer_id=me, type="bug", summary="lỗi")
    feat = await rb.add(customer_id=me, type="new_feature", summary="thêm cột", title="Cột mới")
    items = await rb.list_by_type(me, "new_feature")
    assert [i["id"] for i in items] == [feat.id]
    assert items[0]["title"] == "Cột mới" and items[0]["status"] == "pending"
    # list_bugs is unaffected by the new type
    assert feat.id not in [i.id for i in await rb.list_bugs(me)]


async def test_update_checks_the_type_both_ways():
    rb = RequestBacklog()
    await rb.init()
    me = _cid()
    bug = await rb.add(customer_id=me, type="bug", summary="lỗi")
    feat = await rb.add(customer_id=me, type="new_feature", summary="thêm cột")
    # a feature id on the bug path, and a bug id on the feature path, are both refused
    assert await rb.update_issue(feat.id, me, by="admin", status="completed") is None
    assert (
        await rb.update_issue(bug.id, me, by="admin", type="new_feature", status="processing")
        is None
    )
    updated = await rb.update_issue(feat.id, me, by="admin", type="new_feature", status="responded")
    assert updated is not None and updated.status == "responded"
    # "Đã phản hồi" still waits for the customer: not closed
    assert updated.resolved_at is None
    assert [h.status for h in updated.status_history] == ["responded"]


async def test_updated_feature_is_resolved():
    rb = RequestBacklog()
    await rb.init()
    me = _cid()
    feat = await rb.add(customer_id=me, type="new_feature", summary="thêm cột")
    updated = await rb.update_issue(feat.id, me, by="admin", type="new_feature", status="updated")
    assert updated is not None and updated.resolved_at is not None


async def test_feature_row_with_requester_and_files_round_trips():
    from agent_customer_support.models import RequestFile, Requester

    rb = RequestBacklog()
    await rb.init()
    me = _cid()
    f = RequestFile(name="mẫu.xlsx", media_type="x", s3_key="requests/r/0-m_u.xlsx", size_bytes=5)
    rec = await rb.add(
        id="fixed-" + me,
        customer_id=me,
        type="new_feature",
        status="received",
        summary="thêm cột",
        requester=Requester(name="A", position="TP", department="Thí nghiệm"),
        reason="cần gấp",
        files=[f],
    )
    assert rec.id == "fixed-" + me
    got = await rb.get(rec.id)
    assert got is not None and got.status == "received"
    assert got.requester is not None and got.requester.department == "Thí nghiệm"
    assert got.reason == "cần gấp" and got.files == [f]


async def test_category_is_set_and_cleared():
    rb = RequestBacklog()
    await rb.init()
    me = _cid()
    feat = await rb.add(customer_id=me, type="new_feature", summary="thêm cột")
    up = await rb.update_issue(feat.id, me, by="admin", type="new_feature", category="upgrade")
    assert up is not None and up.category == "upgrade"
    assert up.status_history == []  # not a status change
    up = await rb.update_issue(feat.id, me, by="admin", type="new_feature", category=None)
    assert up is not None and up.category is None


# ---- export reads: created_at range and get_many ---------------------------------------


async def _put_at(rb: RequestBacklog, customer_id: str, at: str, **extra):
    """A row with a chosen `created_at` (stored the way `add` stores it)."""
    import uuid

    from agent_customer_support.models import RequestRecord
    from agent_customer_support.stores.dynamo import get_resource

    rec = RequestRecord(
        id=str(uuid.uuid4()),
        customer_id=customer_id,
        type=extra.pop("type", "bug"),
        summary="s",
        created_at=at,
        **extra,
    )
    async with get_resource() as ddb:
        table = await ddb.Table(rb.table_name)
        await table.put_item(Item=rec.model_dump(mode="json"))
    return rec


async def test_list_by_type_created_range_is_half_open():
    import uuid
    from datetime import UTC, datetime

    rb = RequestBacklog()
    await rb.init()
    cid = f"range-{uuid.uuid4()}"
    before = await _put_at(rb, cid, "2026-11-02T23:59:59.999999Z")
    first = await _put_at(rb, cid, "2026-11-03T00:00:00.000001Z")
    exact = await _put_at(rb, cid, "2026-11-03T00:00:00Z")
    last = await _put_at(rb, cid, "2026-11-04T23:59:59.5Z")
    after = await _put_at(rb, cid, "2026-11-05T00:00:00.25Z")

    lo = datetime(2026, 11, 3, tzinfo=UTC)
    hi = datetime(2026, 11, 5, tzinfo=UTC)
    ids = {i["id"] for i in await rb.list_by_type(cid, "bug", created_from=lo, created_before=hi)}
    assert ids == {first.id, exact.id, last.id}
    assert before.id not in ids and after.id not in ids

    only_lo = {i["id"] for i in await rb.list_by_type(cid, "bug", created_from=lo)}
    assert only_lo == {first.id, exact.id, last.id, after.id}
    only_hi = {i["id"] for i in await rb.list_by_type(cid, "bug", created_before=lo)}
    assert only_hi == {before.id}


async def test_get_many_keeps_order_and_drops_transcript():
    import uuid

    rb = RequestBacklog()
    await rb.init()
    cid = f"many-{uuid.uuid4()}"
    a = await rb.add(customer_id=cid, type="bug", summary="a", transcript="user: dài")
    b = await rb.add(customer_id=cid, type="bug", summary="b", transcript="user: dài")
    got = await rb.get_many([b.id, "missing-id", a.id])
    assert [r.id for r in got] == [b.id, a.id]
    assert all(r.transcript == "" for r in got)


async def test_get_many_more_than_one_batch():
    import uuid

    rb = RequestBacklog()
    await rb.init()
    cid = f"many-{uuid.uuid4()}"
    recs = [
        await _put_at(rb, cid, f"2026-10-01T00:00:{i % 60:02d}Z", note=f"n{i}") for i in range(105)
    ]
    ids = [r.id for r in recs]
    got = await rb.get_many(ids)
    assert [r.id for r in got] == ids
    assert got[104].note == "n104"

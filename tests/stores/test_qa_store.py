import pytest
from agent_customer_support.models import QARecord
from agent_customer_support.stores.qa_store import QAStore

pytestmark = pytest.mark.asyncio


async def test_add_get_update_delete():
    store = QAStore()
    await store.init()
    rec = await store.add(QARecord(question="đổi mật khẩu?", source="manual"))
    got = await store.get(rec.id)
    assert got and got.question == "đổi mật khẩu?"

    got.answer = "Vào Cài đặt > Đổi mật khẩu"
    got.status = "approved"
    await store.update(got)
    again = await store.get(rec.id)
    assert again.answer.startswith("Vào Cài đặt")
    assert again.status == "approved"

    await store.delete(rec.id)
    assert await store.get(rec.id) is None


async def test_list_filters_by_status():
    store = QAStore()
    await store.init()
    p = await store.add(QARecord(question="q-pending", source="manual"))
    a = await store.add(QARecord(question="q-approved", source="manual", status="approved"))
    pending = await store.list(status="pending")
    ids = {r.id for r in pending}
    assert p.id in ids and a.id not in ids


async def test_find_by_feedback_message_id():
    store = QAStore()
    await store.init()
    rec = await store.add(QARecord(question="q", source="feedback", feedback_message_id="msg-123"))
    found = await store.find_by_feedback_message_id("msg-123")
    assert found and found.id == rec.id
    assert await store.find_by_feedback_message_id("nope") is None

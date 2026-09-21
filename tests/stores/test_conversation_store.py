import uuid
import pytest
from agent_customer_support.models import Turn
from agent_customer_support.stores.conversation_store import ConversationStore

pytestmark = pytest.mark.asyncio


async def test_append_and_load():
    cs = ConversationStore()
    await cs.init()
    cid = f"cv-{uuid.uuid4()}"
    await cs.append(cid, "c1", Turn(role="user", content="hi"))
    await cs.append(cid, "c1", Turn(role="assistant", content="hello"))
    conv = await cs.load(cid)
    assert [t.content for t in conv.turns] == ["hi", "hello"]


async def test_load_missing_returns_empty():
    cs = ConversationStore()
    await cs.init()
    conv = await cs.load(f"cv-{uuid.uuid4()}")
    assert conv.customer_id == "" and conv.turns == []


async def test_set_contact_on_existing_conversation_keeps_turns():
    from agent_customer_support.models import ContactInfo

    st = ConversationStore()
    await st.init()
    cid = f"cv-{uuid.uuid4()}"
    await st.append(cid, "c1", Turn(role="user", content="hi"))
    await st.set_contact(cid, "c1", ContactInfo(phone="0912345678", raw="0912345678"))
    conv = await st.load(cid)
    assert conv.contact is not None and conv.contact.phone == "0912345678"
    assert [t.content for t in conv.turns] == ["hi"]
    # a later append keeps the contact
    await st.append(cid, "c1", Turn(role="assistant", content="ok"))
    assert (await st.load(cid)).contact is not None


async def test_set_contact_on_new_conversation_sets_customer():
    from agent_customer_support.models import ContactInfo

    st = ConversationStore()
    cid = f"cv-{uuid.uuid4()}"
    await st.set_contact(cid, "c9", ContactInfo(email="a@b.vn", raw="a@b.vn"))
    conv = await st.load(cid)
    assert conv.customer_id == "c9" and conv.contact and conv.contact.email == "a@b.vn"

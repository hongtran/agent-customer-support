import uuid
import pytest
from agent_customer_support.models import Turn
from agent_customer_support.stores.conversation_store import ConversationStore
pytestmark = pytest.mark.asyncio

async def test_append_and_load():
    cs = ConversationStore(); await cs.init()
    cid = f"cv-{uuid.uuid4()}"
    await cs.append(cid, "c1", Turn(role="user", content="hi"))
    await cs.append(cid, "c1", Turn(role="assistant", content="hello"))
    conv = await cs.load(cid)
    assert [t.content for t in conv.turns] == ["hi", "hello"]

async def test_load_missing_returns_empty():
    cs = ConversationStore(); await cs.init()
    conv = await cs.load(f"cv-{uuid.uuid4()}")
    assert conv.customer_id == "" and conv.turns == []

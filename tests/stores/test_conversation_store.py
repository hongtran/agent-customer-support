import pytest
from agent_customer_support.models import Turn
from agent_customer_support.stores.conversation_store import ConversationStore
pytestmark = pytest.mark.asyncio

async def test_append_and_load():
    cs = ConversationStore(); await cs.init()
    await cs.append("cv1", "c1", Turn(role="user", content="hi"))
    await cs.append("cv1", "c1", Turn(role="assistant", content="hello"))
    conv = await cs.load("cv1")
    assert [t.content for t in conv.turns] == ["hi", "hello"]

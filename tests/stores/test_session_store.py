import pytest
import fakeredis.aioredis
from agent_customer_support.models import SessionState
from agent_customer_support.stores.session_store import SessionStore
pytestmark = pytest.mark.asyncio

async def test_save_and_get():
    r = fakeredis.aioredis.FakeRedis()
    store = SessionStore(client=r)
    await store.save(SessionState(conversation_id="cv1", active_flow_id="f1", current_step_id="s1"))
    got = await store.get("cv1")
    assert got.active_flow_id == "f1" and got.current_step_id == "s1"

async def test_get_missing_returns_fresh():
    r = fakeredis.aioredis.FakeRedis()
    store = SessionStore(client=r)
    got = await store.get("new")
    assert got.conversation_id == "new" and got.active_flow_id is None

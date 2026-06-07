import pytest
from agent_customer_support.stores.request_backlog import RequestBacklog
pytestmark = pytest.mark.asyncio

async def test_add_request():
    rb = RequestBacklog(); await rb.init()
    rec = await rb.add(customer_id="c1", type="feature", summary="thêm cột", module="kinh-doanh", transcript="...")
    assert rec.id and rec.type == "feature"
    got = await rb.get(rec.id)
    assert got and got.summary == "thêm cột"

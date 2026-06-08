import pytest
from agent_customer_support.stores.dynamo import ensure_table, get_resource

pytestmark = pytest.mark.asyncio


async def test_ensure_table_idempotent():
    await ensure_table("acs_smoke", key="id")
    await ensure_table("acs_smoke", key="id")  # second call must not error
    async with get_resource() as ddb:
        table = await ddb.Table("acs_smoke")
        assert (await table.table_status) in {"ACTIVE", "CREATING"}

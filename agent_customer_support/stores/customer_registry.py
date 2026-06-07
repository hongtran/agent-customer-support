from agent_customer_support.config import get_settings
from agent_customer_support.models import CustomerProfile
from agent_customer_support.stores.dynamo import ensure_table, get_resource


class CustomerRegistry:
    def __init__(self) -> None:
        self.table_name = get_settings().table_customers

    async def init(self) -> None:
        await ensure_table(self.table_name, key="customer_id")

    async def put(self, profile: CustomerProfile) -> None:
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            await table.put_item(Item=profile.model_dump(mode="json"))

    async def get(self, customer_id: str) -> CustomerProfile | None:
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            res = await table.get_item(Key={"customer_id": customer_id})
        item = res.get("Item")
        return CustomerProfile.model_validate(item) if item else None

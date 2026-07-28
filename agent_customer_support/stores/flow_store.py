from agent_customer_support.config import get_settings
from agent_customer_support.models import Flow
from agent_customer_support.stores.dynamo import ensure_table, get_resource


class FlowStore:
    def __init__(self) -> None:
        self.table_name = get_settings().table_flows

    async def init(self) -> None:
        await ensure_table(self.table_name, key="id")

    async def upsert(self, flow: Flow) -> None:
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            await table.put_item(Item=flow.model_dump(mode="json"))

    async def get(self, flow_id: str) -> Flow | None:
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            res = await table.get_item(Key={"id": flow_id})
        item = res.get("Item")
        return Flow.model_validate(item) if item else None

    async def list_all(self) -> list[Flow]:
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            res = await table.scan()
        return [Flow.model_validate(i) for i in res.get("Items", [])]

    async def list_for_applications(self, applications: list[str]) -> list[Flow]:
        apps = set(applications)
        return [f for f in await self.list_all() if f.application in apps]

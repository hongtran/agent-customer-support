from boto3.dynamodb.conditions import Attr

from agent_customer_support.config import get_settings
from agent_customer_support.models import QARecord
from agent_customer_support.stores.dynamo import ensure_table, get_resource


class QAStore:
    def __init__(self) -> None:
        self.table_name = get_settings().table_qa

    async def init(self) -> None:
        await ensure_table(self.table_name, key="id")

    async def add(self, record: QARecord) -> QARecord:
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            await table.put_item(Item=record.model_dump(mode="json"))
        return record

    async def get(self, record_id: str) -> QARecord | None:
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            res = await table.get_item(Key={"id": record_id})
        item = res.get("Item")
        return QARecord.model_validate(item) if item else None

    async def list(self, status: str | None = None) -> list[QARecord]:
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            if status:
                res = await table.scan(FilterExpression=Attr("status").eq(status))
            else:
                res = await table.scan()
        return [QARecord.model_validate(i) for i in res.get("Items", [])]

    async def update(self, record: QARecord) -> QARecord:
        from datetime import UTC, datetime

        record.updated_at = datetime.now(UTC)
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            await table.put_item(Item=record.model_dump(mode="json"))
        return record

    async def delete(self, record_id: str) -> None:
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            await table.delete_item(Key={"id": record_id})

    async def find_by_feedback_message_id(self, mid: str) -> QARecord | None:
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            res = await table.scan(FilterExpression=Attr("feedback_message_id").eq(mid))
        items = res.get("Items", [])
        return QARecord.model_validate(items[0]) if items else None

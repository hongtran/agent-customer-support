import uuid
from typing import Literal

from agent_customer_support.config import get_settings
from agent_customer_support.models import RequestRecord
from agent_customer_support.stores.dynamo import ensure_table, get_resource


class RequestBacklog:
    def __init__(self) -> None:
        self.table_name = get_settings().table_requests

    async def init(self) -> None:
        await ensure_table(self.table_name, key="id")

    async def add(
        self,
        *,
        customer_id: str,
        type: Literal["feature", "bug"],
        summary: str,
        application: str | None = None,
        transcript: str = "",
    ) -> RequestRecord:
        rec = RequestRecord(
            id=str(uuid.uuid4()),
            customer_id=customer_id,
            type=type,
            summary=summary,
            application=application,
            transcript=transcript,
        )
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            await table.put_item(Item=rec.model_dump(mode="json"))
        return rec

    async def get(self, request_id: str) -> RequestRecord | None:
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            res = await table.get_item(Key={"id": request_id})
        item = res.get("Item")
        return RequestRecord.model_validate(item) if item else None

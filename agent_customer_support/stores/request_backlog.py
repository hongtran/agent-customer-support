import uuid
from typing import Literal

from agent_customer_support.config import get_settings
from agent_customer_support.models import ContactInfo, RequestRecord
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
        type: Literal["feature", "bug", "how_to_missing"],
        summary: str,
        application: str | None = None,
        transcript: str = "",
        title: str | None = None,
        mantis_issue_id: int | None = None,
        mantis_issue_url: str | None = None,
    ) -> RequestRecord:
        rec = RequestRecord(
            id=str(uuid.uuid4()),
            customer_id=customer_id,
            type=type,
            summary=summary,
            title=title,
            application=application,
            transcript=transcript,
            mantis_issue_id=mantis_issue_id,
            mantis_issue_url=mantis_issue_url,
        )
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            await table.put_item(Item=rec.model_dump(mode="json"))
        return rec

    async def set_contact(self, request_id: str, contact: ContactInfo) -> None:
        """Attach the contact the user gave after the handoff. A single-attribute
        update, not a read-modify-write of the row, so it cannot roll back anything
        written in between."""
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            await table.update_item(
                Key={"id": request_id},
                UpdateExpression="SET contact = :c",
                ExpressionAttributeValues={":c": contact.model_dump(mode="json")},
            )

    async def get(self, request_id: str) -> RequestRecord | None:
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            res = await table.get_item(Key={"id": request_id})
        item = res.get("Item")
        return RequestRecord.model_validate(item) if item else None

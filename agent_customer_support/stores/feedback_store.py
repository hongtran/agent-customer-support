from typing import Any

from boto3.dynamodb.conditions import Attr

from agent_customer_support.config import get_settings
from agent_customer_support.models import FeedbackRecord
from agent_customer_support.stores.dynamo import ensure_table, get_resource


class FeedbackStore:
    """Likes and dislikes on assistant messages, one item per message.

    Kept out of the conversations table on purpose: ``ConversationStore.append`` rewrites
    the whole conversation from a copy read earlier, so a vote stored on a turn could be
    lost to a turn written at the same moment.
    """

    def __init__(self) -> None:
        self.table_name = get_settings().table_feedback

    async def init(self) -> None:
        await ensure_table(self.table_name, key="message_id")

    async def put(self, record: FeedbackRecord) -> FeedbackRecord:
        """Store a vote. Overwrites any earlier vote on the same message."""
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            await table.put_item(Item=record.model_dump(mode="json"))
        return record

    async def get(self, message_id: str) -> FeedbackRecord | None:
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            res = await table.get_item(Key={"message_id": message_id})
        item = res.get("Item")
        return FeedbackRecord.model_validate(item) if item else None

    async def delete(self, message_id: str) -> None:
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            await table.delete_item(Key={"message_id": message_id})

    async def list(
        self, signal: str | None = None, customer_id: str | None = None
    ) -> list[FeedbackRecord]:
        """All votes matching the filters, newest first.

        A scan reads at most 1 MB per call, so this follows ``LastEvaluatedKey`` —
        without it a large table would silently return only the first page.
        """
        cond = None
        if signal:
            cond = Attr("signal").eq(signal)
        if customer_id:
            by_customer = Attr("customer_id").eq(customer_id)
            cond = by_customer if cond is None else cond & by_customer
        kwargs: dict[str, Any] = {}
        if cond is not None:
            kwargs["FilterExpression"] = cond
        items: list[dict] = []
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            while True:
                res = await table.scan(**kwargs)
                items.extend(res.get("Items", []))
                last = res.get("LastEvaluatedKey")
                if not last:
                    break
                kwargs["ExclusiveStartKey"] = last
        records = [FeedbackRecord.model_validate(i) for i in items]
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records

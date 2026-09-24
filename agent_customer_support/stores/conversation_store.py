import base64
import json
from typing import Any

from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

from agent_customer_support.config import get_settings
from agent_customer_support.stores.dynamo import get_resource
from agent_customer_support.models import Conversation, ConversationSummary, Turn, ContactInfo

# Lists one customer's conversations, newest first, without scanning the table. Sparse:
# a row with no `updated_at` (written before this index existed) is not in it until
# scripts/backfill_conversation_summary.py rewrites it.
BY_CUSTOMER_INDEX = "customer_id-updated_at-index"

_INDEX_ATTRS = [
    {"AttributeName": "customer_id", "AttributeType": "S"},
    {"AttributeName": "updated_at", "AttributeType": "S"},
]
_INDEX = {
    "IndexName": BY_CUSTOMER_INDEX,
    "KeySchema": [
        {"AttributeName": "customer_id", "KeyType": "HASH"},
        {"AttributeName": "updated_at", "KeyType": "RANGE"},
    ],
    # Only what a list row shows. Projecting ALL would copy every transcript a second
    # time, which is what makes a conversation item large in the first place.
    "Projection": {
        "ProjectionType": "INCLUDE",
        "NonKeyAttributes": ["title", "created_at", "turn_count"],
    },
}


def _encode_cursor(key: dict[str, Any]) -> str:
    return base64.urlsafe_b64encode(json.dumps(key).encode()).decode()


def _decode_cursor(cursor: str) -> dict[str, Any]:
    return json.loads(base64.urlsafe_b64decode(cursor.encode()))


class InvalidCursorError(ValueError):
    pass


class ConversationStore:
    def __init__(self) -> None:
        self.table_name = get_settings().table_conversations

    async def init(self) -> None:
        """Create the table with its by-customer index, or add the index to a table that
        predates it. Guarded like `ensure_table`: with auto-create off (prod), the table
        and the index are infrastructure's job."""
        if not get_settings().dynamodb_auto_create_tables:
            return
        async with get_resource() as ddb:
            client = ddb.meta.client
            try:
                await client.create_table(
                    TableName=self.table_name,
                    KeySchema=[{"AttributeName": "conversation_id", "KeyType": "HASH"}],
                    AttributeDefinitions=[
                        {"AttributeName": "conversation_id", "AttributeType": "S"},
                        *_INDEX_ATTRS,
                    ],
                    GlobalSecondaryIndexes=[_INDEX],
                    BillingMode="PAY_PER_REQUEST",
                )
                return
            except ClientError as e:
                if e.response["Error"]["Code"] != "ResourceInUseException":
                    raise
            desc = await client.describe_table(TableName=self.table_name)
            existing = {i["IndexName"] for i in desc["Table"].get("GlobalSecondaryIndexes", [])}
            if BY_CUSTOMER_INDEX in existing:
                return
            await client.update_table(
                TableName=self.table_name,
                AttributeDefinitions=_INDEX_ATTRS,
                GlobalSecondaryIndexUpdates=[{"Create": _INDEX}],
            )

    async def load(self, conversation_id: str) -> Conversation:
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            res = await table.get_item(Key={"conversation_id": conversation_id})
        item = res.get("Item")
        if not item:
            return Conversation(conversation_id=conversation_id, customer_id="")
        return Conversation.model_validate(item)

    async def append(self, conversation_id: str, customer_id: str, turn: Turn) -> None:
        conv = await self.load(conversation_id)
        if not conv.customer_id:
            conv.customer_id = customer_id
        conv.turns.append(turn)
        await self.put(conv)

    async def put(self, conv: Conversation) -> None:
        """Write the whole item. Refreshes the summary first, so every write keeps the
        by-customer index current and never stores a null index key."""
        conv.refresh_summary()
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            await table.put_item(Item=conv.model_dump(mode="json"))

    async def set_contact(
        self, conversation_id: str, customer_id: str, contact: ContactInfo
    ) -> None:
        """Record the contact the user left after a handoff. Same load-modify-put as
        `append`, so a row that does not exist yet is created with its customer."""
        conv = await self.load(conversation_id)
        if not conv.customer_id:
            conv.customer_id = customer_id
        conv.contact = contact
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            await table.put_item(Item=conv.model_dump(mode="json"))

    async def list_by_customer(
        self, customer_id: str, limit: int = 50, cursor: str | None = None
    ) -> tuple[list[ConversationSummary], str | None]:
        """One page of a customer's conversations, most recently active first.

        Returns the page and an opaque cursor for the next one (None on the last page).
        Raises InvalidCursorError for a cursor this method did not produce.
        """
        kwargs: dict[str, Any] = {
            "IndexName": BY_CUSTOMER_INDEX,
            "KeyConditionExpression": Key("customer_id").eq(customer_id),
            "ScanIndexForward": False,
            "Limit": limit,
        }
        if cursor:
            try:
                start = _decode_cursor(cursor)
            except (ValueError, TypeError) as exc:
                raise InvalidCursorError(cursor) from exc
            # A cursor minted for another customer would page through their index.
            if not isinstance(start, dict) or start.get("customer_id") != customer_id:
                raise InvalidCursorError(cursor)
            kwargs["ExclusiveStartKey"] = start
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            res = await table.query(**kwargs)
        items = [ConversationSummary.model_validate(i) for i in res.get("Items", [])]
        last = res.get("LastEvaluatedKey")
        return items, (_encode_cursor(last) if last else None)

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from botocore.exceptions import ClientError

from agent_customer_support.config import get_settings
from agent_customer_support.models import CustomerProfile
from agent_customer_support.stores.dynamo import ensure_table, get_resource


def today() -> date:
    """The current day in the configured timezone. Using UTC would roll the limit over
    at 07:00 Vietnam time, in the middle of a working morning."""
    return datetime.now(ZoneInfo(get_settings().usage_timezone)).date()


class UsageStore:
    """Per-customer, per-day question counter.

    One item per customer per day, keyed ``<customer_id>#<YYYY-MM-DD>``. The date in the
    key is the whole reset mechanism: at midnight the key changes, the new key has no
    item, and the count starts from zero. There is no reset job that could fail or run
    twice. Old items are removed by DynamoDB TTL on ``expires_at``.

    Kept out of the customers table on purpose: the admin PATCH rewrites the whole
    profile item from a copy read earlier, so a counter stored there would be rolled
    back by any concurrent profile edit.
    """

    def __init__(self) -> None:
        self.table_name = get_settings().table_usage

    async def init(self) -> None:
        await ensure_table(self.table_name, key="id")
        if not get_settings().dynamodb_auto_create_tables:
            return
        # TTL is what deletes old days. Enabling it twice is a ValidationException, which
        # just means it is already on.
        async with get_resource() as ddb:
            client = ddb.meta.client
            try:
                await client.update_time_to_live(
                    TableName=self.table_name,
                    TimeToLiveSpecification={"Enabled": True, "AttributeName": "expires_at"},
                )
            except ClientError as e:
                if e.response["Error"]["Code"] != "ValidationException":
                    raise

    @staticmethod
    def _key(customer_id: str, day: date) -> str:
        return f"{customer_id}#{day.isoformat()}"

    async def try_consume(self, customer_id: str, limit: int) -> int | None:
        """Count one question if the customer is under ``limit`` today.

        Check and increment are a single conditional UpdateItem, so concurrent requests
        cannot both take the last slot. Returns today's count after this question, or
        None when the limit is reached. The count comes back from the same write
        (UPDATED_NEW), so reporting "remaining" costs no extra read.
        """
        # The condition below lets a missing item through, so a limit of 0 (a blocked
        # customer) would still allow the first question of each day without this.
        if limit <= 0:
            return None
        s = get_settings()
        day = today()
        tz = ZoneInfo(s.usage_timezone)
        expires = datetime.combine(day + timedelta(days=s.usage_retention_days), time(), tz)
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            try:
                res = await table.update_item(
                    Key={"id": self._key(customer_id, day)},
                    UpdateExpression=(
                        "SET #count = if_not_exists(#count, :zero) + :one, "
                        "customer_id = :cid, #date = :date, expires_at = :exp"
                    ),
                    ConditionExpression="attribute_not_exists(#count) OR #count < :limit",
                    ExpressionAttributeNames={"#count": "count", "#date": "date"},
                    ExpressionAttributeValues={
                        ":zero": 0,
                        ":one": 1,
                        ":limit": limit,
                        ":cid": customer_id,
                        ":date": day.isoformat(),
                        ":exp": int(expires.timestamp()),
                    },
                    ReturnValues="UPDATED_NEW",
                )
            except ClientError as e:
                if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                    return None
                raise
        return int(res["Attributes"]["count"])

    async def get_today(self, customer_id: str) -> int:
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            res = await table.get_item(Key={"id": self._key(customer_id, today())})
        item = res.get("Item")
        return int(item["count"]) if item else 0

    async def remaining_today(self, customer: CustomerProfile) -> int | None:
        """Questions the customer can still ask today; None means unlimited.

        Admins and customers without a limit return None without touching the table.
        """
        limit = customer.daily_question_limit
        if customer.role == "admin" or limit is None:
            return None
        return max(0, limit - await self.get_today(customer.customer_id))

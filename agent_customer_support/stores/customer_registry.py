from botocore.exceptions import ClientError

from agent_customer_support.config import get_settings
from agent_customer_support.models import CustomerProfile
from agent_customer_support.stores.dynamo import ensure_table, get_resource


class CustomerExistsError(Exception):
    """Raised by create() when the customer_id is already taken."""


class CustomerRegistry:
    def __init__(self) -> None:
        self.table_name = get_settings().table_customers

    async def init(self) -> None:
        await ensure_table(self.table_name, key="customer_id")

    async def put(self, profile: CustomerProfile) -> None:
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            await table.put_item(Item=profile.model_dump(mode="json"))

    async def create(self, profile: CustomerProfile) -> None:
        """Insert a new profile, refusing to overwrite an existing one.

        put_item is an upsert, and customer_id is the login username — so a plain put
        here would let an admin creating "ttp" silently replace a live tenant's profile
        and hand their id, conversations and application scope to the new account. The
        condition makes that a 409 instead.
        """
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            try:
                await table.put_item(
                    Item=profile.model_dump(mode="json"),
                    ConditionExpression="attribute_not_exists(customer_id)",
                )
            except ClientError as e:
                if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                    raise CustomerExistsError(profile.customer_id) from e
                raise

    async def get(self, customer_id: str) -> CustomerProfile | None:
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            res = await table.get_item(Key={"customer_id": customer_id})
        item = res.get("Item")
        return CustomerProfile.model_validate(item) if item else None

    async def list(self) -> list[CustomerProfile]:
        """All profiles. A scan, which is fine here: the table holds tens of tenants
        and this backs an admin-only screen, not a request path."""
        items: list[dict] = []
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            kwargs: dict = {}
            while True:
                res = await table.scan(**kwargs)
                items.extend(res.get("Items", []))
                key = res.get("LastEvaluatedKey")
                if not key:
                    break
                kwargs["ExclusiveStartKey"] = key
        return [CustomerProfile.model_validate(i) for i in items]

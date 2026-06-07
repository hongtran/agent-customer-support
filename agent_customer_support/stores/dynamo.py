import contextlib
import aioboto3
from botocore.exceptions import ClientError
from agent_customer_support.config import get_settings

def _session() -> aioboto3.Session:
    return aioboto3.Session()

@contextlib.asynccontextmanager
async def get_resource():
    s = get_settings()
    async with _session().resource(
        "dynamodb",
        endpoint_url=s.dynamodb_endpoint_url,
        region_name=s.aws_region,
    ) as ddb:
        yield ddb

async def ensure_table(name: str, key: str = "id") -> None:
    async with get_resource() as ddb:
        try:
            await ddb.create_table(
                TableName=name,
                KeySchema=[{"AttributeName": key, "KeyType": "HASH"}],
                AttributeDefinitions=[{"AttributeName": key, "AttributeType": "S"}],
                BillingMode="PAY_PER_REQUEST",
            )
        except ClientError as e:
            if e.response["Error"]["Code"] != "ResourceInUseException":
                raise

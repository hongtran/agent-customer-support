import pytest
from agent_customer_support.models import CustomerProfile
from agent_customer_support.stores.customer_registry import CustomerRegistry
pytestmark = pytest.mark.asyncio

async def test_put_get_customer():
    reg = CustomerRegistry(); await reg.init()
    await reg.put(CustomerProfile(customer_id="c1", name="C1", enabled_modules=["xet-nghiem"]))
    got = await reg.get("c1")
    assert got and got.enabled_modules == ["xet-nghiem"]

async def test_get_missing_returns_none():
    reg = CustomerRegistry(); await reg.init()
    assert await reg.get("nope") is None

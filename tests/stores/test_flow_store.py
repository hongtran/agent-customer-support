import pytest
from agent_customer_support.models import Flow, FlowStep, FlowTransition, FlowOutcome
from agent_customer_support.stores.flow_store import FlowStore

pytestmark = pytest.mark.asyncio

def _flow(fid="f1", module="xet-nghiem", scope="global"):
    return Flow(
        id=fid, title="t", module=module, scope=scope, version=1, language="vi",
        triggers=["tạo mẫu"],
        steps=[FlowStep(id="s1", say="hi", next=[FlowTransition(when="ok", goto="done")])],
        outcomes={"done": FlowOutcome(type="success", say="bye")},
    )

async def test_import_and_get():
    store = FlowStore()
    await store.init()
    await store.upsert(_flow("fA"))
    got = await store.get("fA")
    assert got is not None and got.id == "fA"

async def test_list_for_customer_filters_by_module():
    store = FlowStore()
    await store.init()
    await store.upsert(_flow("fX", module="xet-nghiem"))
    await store.upsert(_flow("fQ", module="quan-trac"))
    flows = await store.list_for_modules(["xet-nghiem"])
    ids = {f.id for f in flows}
    assert "fX" in ids and "fQ" not in ids

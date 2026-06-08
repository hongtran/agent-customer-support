import asyncio, json, pathlib, sys
from agent_customer_support.models import Flow
from agent_customer_support.stores.flow_store import FlowStore

async def main(folder: str) -> None:
    store = FlowStore()
    await store.init()
    for p in pathlib.Path(folder).glob("*.json"):
        flow = Flow.model_validate(json.loads(p.read_text()))
        await store.upsert(flow)
        print(f"imported {flow.id} ({p.name})")

if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "seeds/flows"))

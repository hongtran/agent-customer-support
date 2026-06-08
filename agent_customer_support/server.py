import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from agent_customer_support.channels.widget import router as widget_router, get_agent
from agent_customer_support.stores.customer_registry import CustomerRegistry
from agent_customer_support.stores.conversation_store import ConversationStore
from agent_customer_support.stores.flow_store import FlowStore
from agent_customer_support.stores.request_backlog import RequestBacklog

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    for store in (CustomerRegistry(), ConversationStore(), FlowStore(), RequestBacklog()):
        try:
            await store.init()
        except Exception as exc:
            logger.warning("Store init skipped (%s): %s", type(store).__name__, exc)
    yield


app = FastAPI(title="CenLab Support Agent", lifespan=lifespan)
app.include_router(widget_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# re-export get_agent so tests can use app.dependency_overrides[get_agent]
__all__ = ["app", "get_agent"]

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from agent_customer_support.channels.widget import router as widget_router, get_agent
from agent_customer_support.channels.admin import router as admin_router
from agent_customer_support.config import get_settings
from agent_customer_support.stores.qa_store import QAStore
from agent_customer_support.observability import tracing
from agent_customer_support.stores.customer_registry import CustomerRegistry
from agent_customer_support.stores.conversation_store import ConversationStore
from agent_customer_support.stores.flow_store import FlowStore
from agent_customer_support.stores.request_backlog import RequestBacklog

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    for store in (CustomerRegistry(), ConversationStore(), FlowStore(), RequestBacklog(), QAStore()):
        try:
            await store.init()
        except Exception as exc:
            logger.warning("Store init skipped (%s): %s", type(store).__name__, exc)
    yield
    # Flush any buffered traces on shutdown (no-op when tracing is disabled).
    tracing.flush()


app = FastAPI(title="CenLab Support Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["POST", "GET", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["Content-Type", "X-Admin-Token"],
)

app.include_router(widget_router)
app.include_router(admin_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# re-export get_agent so tests can use app.dependency_overrides[get_agent]
__all__ = ["app", "get_agent"]

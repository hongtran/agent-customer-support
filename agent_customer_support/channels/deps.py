from functools import lru_cache

from fastapi import Header, HTTPException

from agent_customer_support.config import get_settings
from agent_customer_support.rag.qa_indexer import QAIndexer
from agent_customer_support.stores.conversation_store import ConversationStore
from agent_customer_support.stores.qa_store import QAStore


@lru_cache
def get_qa_store() -> QAStore:
    return QAStore()


@lru_cache
def get_conversation_store() -> ConversationStore:
    return ConversationStore()


@lru_cache
def get_qa_indexer() -> QAIndexer:
    return QAIndexer()


def require_admin(x_admin_token: str = Header(default="")) -> None:
    token = get_settings().admin_token
    if not token or x_admin_token != token:
        raise HTTPException(status_code=401, detail="invalid admin token")

from functools import lru_cache

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from agent_customer_support.auth import decode_access_token
from agent_customer_support.models import CustomerProfile
from agent_customer_support.rag.qa_indexer import QAIndexer
from agent_customer_support.stores.conversation_store import ConversationStore
from agent_customer_support.stores.customer_registry import CustomerRegistry
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


@lru_cache
def get_customer_registry() -> CustomerRegistry:
    return CustomerRegistry()


# auto_error=False so a missing header reaches our own handler and returns the same
# shape as a bad one.
_bearer = HTTPBearer(auto_error=False)

_UNAUTHORIZED = HTTPException(
    status_code=401,
    detail="not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)


async def get_current_customer(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    registry: CustomerRegistry = Depends(get_customer_registry),
) -> CustomerProfile:
    """Resolve the caller from the access token.

    The profile is re-read from the registry rather than reconstructed from the token's
    claims: the token is stateless and unrevocable, so this read is what makes a deleted
    customer or a demoted admin take effect immediately instead of at token expiry. It
    costs one DynamoDB get against an LLM turn that costs far more.
    """
    if creds is None or not creds.credentials:
        raise _UNAUTHORIZED
    try:
        payload = decode_access_token(creds.credentials)
    except jwt.PyJWTError as exc:
        raise _UNAUTHORIZED from exc
    customer_id = payload.get("sub")
    if not customer_id:
        raise _UNAUTHORIZED
    profile = await registry.get(customer_id)
    if profile is None:
        raise _UNAUTHORIZED
    return profile


def require_admin(
    customer: CustomerProfile = Depends(get_current_customer),
) -> CustomerProfile:
    if customer.role != "admin":
        raise HTTPException(status_code=403, detail="admin role required")
    return customer

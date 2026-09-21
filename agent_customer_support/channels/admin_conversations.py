from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from agent_customer_support import image_urls
from agent_customer_support.channels.deps import (
    get_attachment_store,
    get_conversation_store,
    get_customer_registry,
    get_doc_image_store,
    require_admin,
)
from agent_customer_support.models import AttachmentRef, ConversationSummary
from agent_customer_support.stores.attachment_store import AttachmentStore
from agent_customer_support.stores.conversation_store import (
    ConversationStore,
    InvalidCursorError,
)
from agent_customer_support.stores.customer_registry import CustomerRegistry
from agent_customer_support.stores.doc_image_store import DocImageStore

router = APIRouter(
    prefix="/admin/customers/{customer_id}/conversations",
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)


class ConversationPage(BaseModel):
    items: list[ConversationSummary]
    next_cursor: str | None = None


class TurnOut(BaseModel):
    """A persisted turn rendered for reading: image keys and markers become URLs."""

    id: str
    role: Literal["user", "assistant"]
    content: str
    ts: datetime
    attachments: list[AttachmentRef]


class ConversationDetail(BaseModel):
    conversation_id: str
    customer_id: str
    turns: list[TurnOut]


@router.get("")
async def list_conversations(
    customer_id: str,
    limit: int = Query(default=50, ge=1, le=100),
    cursor: str | None = None,
    registry: CustomerRegistry = Depends(get_customer_registry),
    convs: ConversationStore = Depends(get_conversation_store),
) -> ConversationPage:
    """A customer's conversations, most recently active first, one page at a time."""
    if await registry.get(customer_id) is None:
        raise HTTPException(status_code=404, detail="customer not found")
    try:
        items, next_cursor = await convs.list_by_customer(customer_id, limit, cursor)
    except InvalidCursorError as exc:
        raise HTTPException(status_code=400, detail="invalid cursor") from exc
    return ConversationPage(items=items, next_cursor=next_cursor)


@router.get("/{conversation_id}")
async def get_conversation(
    customer_id: str,
    conversation_id: str,
    convs: ConversationStore = Depends(get_conversation_store),
    attachments: AttachmentStore = Depends(get_attachment_store),
    doc_images: DocImageStore = Depends(get_doc_image_store),
) -> ConversationDetail:
    """Every turn of one conversation, with images signed for display.

    The conversation must belong to the customer in the path. The path is admin-chosen,
    so this is not a tenant boundary, but a mismatched pair would show one customer's
    chat on another customer's page — 404 keeps the page honest.
    """
    conv = await convs.load(conversation_id)
    if not conv.turns or conv.customer_id != customer_id:
        raise HTTPException(status_code=404, detail="conversation not found")
    turns = []
    for t in conv.turns:
        if t.role == "assistant":
            content = await image_urls.resolve_doc_images(doc_images, t.content)
            refs: list[AttachmentRef] = []
        else:
            content = t.content
            refs = await image_urls.presign_attachments(attachments, t.attachments)
        turns.append(TurnOut(id=t.id, role=t.role, content=content, ts=t.ts, attachments=refs))
    return ConversationDetail(
        conversation_id=conv.conversation_id, customer_id=conv.customer_id, turns=turns
    )

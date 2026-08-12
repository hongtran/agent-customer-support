from functools import lru_cache
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from agent_customer_support.models import (
    ChatRequest,
    ChatResponse,
    Conversation,
    CustomerProfile,
    QARecord,
)
from agent_customer_support.agents.coordinator import Coordinator
from agent_customer_support.channels.deps import (
    get_conversation_store,
    get_current_customer,
    get_qa_store,
)
from agent_customer_support.config import get_settings
from agent_customer_support.stores.conversation_store import ConversationStore
from agent_customer_support.stores.qa_store import QAStore

router = APIRouter(prefix="/widget", tags=["widget"])


@lru_cache
def get_agent() -> Coordinator:
    return Coordinator()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    agent: Coordinator = Depends(get_agent),
    customer: CustomerProfile = Depends(get_current_customer),
) -> ChatResponse:
    # Size-check before anything else: an oversized upload should cost nothing, and
    # everything downstream (S3, the LLM) is more expensive than this comparison.
    # decoded_size is arithmetic on the base64 length, so the payload is never
    # materialised just to measure it.
    total = sum(a.decoded_size for a in req.attachments)
    limit = get_settings().max_attachment_bytes
    if total > limit:
        raise HTTPException(
            status_code=413,
            detail=f"attachments total {total} bytes, limit is {limit}",
        )
    return await agent.handle_turn(
        # From the token, never the body — this is what makes the tenant boundary real.
        customer_id=customer.customer_id,
        conversation_id=req.conversation_id,
        message=req.message,
        attachments=req.attachments,
        applications=req.applications or None,
    )


class CustomerApplicationsResponse(BaseModel):
    customer_id: str
    enabled_applications: list[str]


@router.get("/me/applications", response_model=CustomerApplicationsResponse)
async def get_my_applications(
    customer: CustomerProfile = Depends(get_current_customer),
) -> CustomerApplicationsResponse:
    """Applications enabled for the caller.

    Deliberately has no path parameter: the old /widget/customer/{id} let anyone read
    any tenant's application list. Taking the id from the token makes a cross-tenant
    read unrepresentable rather than merely rejected.
    """
    return CustomerApplicationsResponse(
        customer_id=customer.customer_id,
        enabled_applications=customer.enabled_applications,
    )


class FeedbackRequest(BaseModel):
    conversation_id: str
    message_id: str
    signal: Literal["down"] = "down"


def _transcript(conv: Conversation) -> str:
    return "\n".join(f"{t.role}: {t.content}" for t in conv.turns)


@router.post("/feedback")
async def feedback(
    req: FeedbackRequest,
    qa: QAStore = Depends(get_qa_store),
    convs: ConversationStore = Depends(get_conversation_store),
    customer: CustomerProfile = Depends(get_current_customer),
) -> dict:
    conv = await convs.load(req.conversation_id)
    # Ownership check before anything is read out of the conversation: a QARecord copies
    # the full transcript, so downvoting someone else's conversation would have pulled
    # their messages into the Q&A store. 404 rather than 403 — a conversation you don't
    # own shouldn't be confirmed to exist.
    if conv.customer_id and conv.customer_id != customer.customer_id:
        raise HTTPException(status_code=404, detail="conversation not found")
    idx = next(
        (i for i, t in enumerate(conv.turns) if t.id == req.message_id and t.role == "assistant"),
        None,
    )
    if idx is None:
        raise HTTPException(status_code=404, detail="message not found")
    bad_answer = conv.turns[idx].content
    question = next(
        (conv.turns[j].content for j in range(idx - 1, -1, -1) if conv.turns[j].role == "user"),
        "",
    )
    existing = await qa.find_by_feedback_message_id(req.message_id)
    if existing:
        existing.question = question
        existing.bad_answer = bad_answer
        await qa.update(existing)
        return {"ok": True}
    await qa.add(
        QARecord(
            question=question,
            source="feedback",
            status="pending",
            bad_answer=bad_answer,
            customer_id=customer.customer_id,
            conversation_id=req.conversation_id,
            feedback_message_id=req.message_id,
            transcript=_transcript(conv),
        )
    )
    return {"ok": True}

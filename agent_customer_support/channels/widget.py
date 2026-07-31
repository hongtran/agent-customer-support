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
from agent_customer_support.stores.customer_registry import CustomerRegistry
from agent_customer_support.channels.deps import get_conversation_store, get_qa_store
from agent_customer_support.config import get_settings
from agent_customer_support.stores.conversation_store import ConversationStore
from agent_customer_support.stores.qa_store import QAStore

router = APIRouter(prefix="/widget", tags=["widget"])


@lru_cache
def get_agent() -> Coordinator:
    return Coordinator()


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, agent: Coordinator = Depends(get_agent)) -> ChatResponse:
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
        customer_id=req.customer_id,
        conversation_id=req.conversation_id,
        message=req.message,
        attachments=req.attachments,
        applications=req.applications or None,
    )


class CustomerApplicationsResponse(BaseModel):
    customer_id: str
    enabled_applications: list[str]


@router.get("/customer/{customer_id}", response_model=CustomerApplicationsResponse)
async def get_customer_applications(customer_id: str) -> CustomerApplicationsResponse:
    registry = CustomerRegistry()
    profile: CustomerProfile | None = await registry.get(customer_id)
    applications = profile.enabled_applications if profile else []
    return CustomerApplicationsResponse(customer_id=customer_id, enabled_applications=applications)


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
) -> dict:
    conv = await convs.load(req.conversation_id)
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
            customer_id=conv.customer_id or None,
            conversation_id=req.conversation_id,
            feedback_message_id=req.message_id,
            transcript=_transcript(conv),
        )
    )
    return {"ok": True}

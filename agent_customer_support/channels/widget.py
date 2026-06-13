from fastapi import APIRouter, Depends
from agent_customer_support.models import ChatRequest, ChatResponse
from agent_customer_support.agents.coordinator import Coordinator

router = APIRouter(prefix="/widget", tags=["widget"])


def get_agent() -> Coordinator:
    return Coordinator()


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, agent: Coordinator = Depends(get_agent)) -> ChatResponse:
    return await agent.handle_turn(
        customer_id=req.customer_id,
        conversation_id=req.conversation_id,
        message=req.message,
        attachments=req.attachments,
    )

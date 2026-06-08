from fastapi import APIRouter, Depends
from agent_customer_support.models import ChatRequest, ChatResponse
from agent_customer_support.agent.core import AgentCore

router = APIRouter(prefix="/widget", tags=["widget"])


def get_agent() -> AgentCore:
    return AgentCore()


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, agent: AgentCore = Depends(get_agent)) -> ChatResponse:
    return await agent.handle_turn(
        customer_id=req.customer_id,
        conversation_id=req.conversation_id,
        user_msg=req.message,
    )

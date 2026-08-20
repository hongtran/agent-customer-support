import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_customer_support.agents.knowledge import KnowledgeAgent
from agent_customer_support.agents.context import TurnContext
from agent_customer_support.models import CustomerProfile, SessionState, Conversation

pytestmark = pytest.mark.asyncio


async def test_confirmed_miss_creates_pending_qa_record(monkeypatch):
    agent = KnowledgeAgent()

    # Force the "second miss after clarification" path: compose returns a miss marker.
    monkeypatch.setattr(agent, "_contextualize", AsyncMock(return_value="câu hỏi lạ"))
    monkeypatch.setattr(agent, "_compose", AsyncMock(return_value="[[no_answer]]"))

    session = SessionState(conversation_id="c1", pending="knowledge_clarify")
    ctx = TurnContext(
        customer=CustomerProfile(customer_id="cust1", name="N"),
        session=session,
        conversation=Conversation(conversation_id="c1", customer_id="cust1"),
        message="câu hỏi lạ",
        transcript="assistant: ...\nuser: câu hỏi lạ",
    )
    ctx.rag = MagicMock()
    ctx.rag.search = AsyncMock(return_value={"passages": [], "citations": []})
    ctx.rag.search_with_fallback = ctx.rag.search  # product search entry point
    ctx.backlog = MagicMock()
    ctx.backlog.add = AsyncMock()
    ctx.qa_store = MagicMock()
    ctx.qa_store.add = AsyncMock()

    res = await agent.run(ctx)

    assert res.resolved is False
    ctx.qa_store.add.assert_awaited_once()
    rec = ctx.qa_store.add.await_args.args[0]
    assert rec.source == "cannot_answer"
    assert rec.status == "pending"
    assert rec.question == "câu hỏi lạ"
    assert rec.conversation_id == "c1"

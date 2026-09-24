import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_customer_support.agents.knowledge import KnowledgeAgent
from agent_customer_support.agents.context import TurnContext
from tests.agents.composed import composed_answer
from agent_customer_support.models import CustomerProfile, SessionState, Conversation

pytestmark = pytest.mark.asyncio


async def test_confirmed_miss_creates_pending_qa_record(monkeypatch):
    agent = KnowledgeAgent()

    # Force the "gave up" path: a miss with no clarifying budget left.
    monkeypatch.setattr(agent, "_contextualize", AsyncMock(return_value="câu hỏi lạ"))
    monkeypatch.setattr(
        agent,
        "_compose",
        AsyncMock(return_value=composed_answer(answer="[[no_answer]]", cited=[])),
    )

    session = SessionState(conversation_id="c1", clarify_count=2)
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

    res = await agent.run(ctx, allow_clarify=False)

    assert res.knowledge_status == "no_answer"
    ctx.qa_store.add.assert_awaited_once()
    rec = ctx.qa_store.add.await_args.args[0]
    assert rec.source == "cannot_answer"
    assert rec.status == "pending"
    assert rec.question == "câu hỏi lạ"
    assert rec.conversation_id == "c1"

import pytest
from unittest.mock import AsyncMock
from agent_customer_support.agents.knowledge import KnowledgeAgent
from agent_customer_support.agents.context import TurnContext
from agent_customer_support.config import get_settings
from agent_customer_support.models import CustomerProfile, SessionState, Conversation

pytestmark = pytest.mark.asyncio


def _ctx(applications=None):
    session = SessionState(conversation_id="c1", selected_applications=applications or [])
    ctx = TurnContext(
        customer=CustomerProfile(customer_id="cust1", name="N"),
        session=session,
        conversation=Conversation(conversation_id="c1", customer_id="cust1"),
        message="làm sao tạo phiếu?",
        transcript="",
    )
    return ctx


def _patch_compose(monkeypatch, agent):
    cap = {}

    async def fake_compose(
        question, passages, transcript, cfg, allow_clarify=True, qa_passages=None, qa_leads=False
    ):
        cap["passages"] = passages
        cap["qa_passages"] = qa_passages
        cap["qa_leads"] = qa_leads
        return "Anh/Chị vui lòng làm theo hướng dẫn."  # plain answer, no marker

    monkeypatch.setattr(agent, "_compose", fake_compose)
    monkeypatch.setattr(agent, "_contextualize", AsyncMock(return_value="q-standalone"))
    return cap


def _search_dispatch(qa_result, product_result=None):
    cfg = get_settings()
    product_result = product_result or {
        "passages": ["guide"],
        "citations": ["g1"],
        "top_confidence": 0.5,
    }

    async def search(query, collection, applications=None, **kwargs):
        if collection == cfg.qa_collection:
            if isinstance(qa_result, Exception):
                raise qa_result
            return qa_result
        return product_result

    return search


async def test_qa_leads_when_above_threshold(monkeypatch):
    agent = KnowledgeAgent()
    cap = _patch_compose(monkeypatch, agent)
    ctx = _ctx()
    ctx.rag = type("R", (), {})()
    ctx.rag.search = AsyncMock(
        side_effect=_search_dispatch(
            {"passages": ["cs answer"], "citations": ["abc"], "top_confidence": 0.95}
        )
    )
    res = await agent.run(ctx)
    assert cap["qa_passages"] == ["cs answer"]
    assert cap["qa_leads"] is True
    assert "qa:abc" in res.citations
    assert "g1" in res.citations


async def test_qa_supplementary_when_below_threshold(monkeypatch):
    agent = KnowledgeAgent()
    cap = _patch_compose(monkeypatch, agent)
    ctx = _ctx()
    ctx.rag = type("R", (), {})()
    ctx.rag.search = AsyncMock(
        side_effect=_search_dispatch(
            {"passages": ["cs answer"], "citations": ["abc"], "top_confidence": 0.4}
        )
    )
    await agent.run(ctx)
    assert cap["qa_passages"] == ["cs answer"]
    assert cap["qa_leads"] is False


async def test_qa_search_failure_degrades_to_product_only(monkeypatch):
    agent = KnowledgeAgent()
    cap = _patch_compose(monkeypatch, agent)
    ctx = _ctx()
    ctx.rag = type("R", (), {})()
    # ValueError is what Qdrant local mode raises for a missing collection; the
    # narrowed except in _safe_qa_search catches store failures, not RuntimeError.
    ctx.rag.search = AsyncMock(side_effect=_search_dispatch(ValueError("no collection")))
    await agent.run(ctx)  # must not raise
    assert cap["qa_passages"] == []
    assert cap["qa_leads"] is False
    assert cap["passages"] == ["guide"]


async def test_applications_filter_passed_to_qa_search(monkeypatch):
    agent = KnowledgeAgent()
    _patch_compose(monkeypatch, agent)
    ctx = _ctx(applications=["Lab"])
    ctx.rag = type("R", (), {})()
    ctx.rag.search = AsyncMock(
        side_effect=_search_dispatch({"passages": [], "citations": [], "top_confidence": 0.0})
    )
    await agent.run(ctx)
    cfg = get_settings()
    ctx.rag.search.assert_any_await(
        "q-standalone",
        collection=cfg.qa_collection,
        applications=["Lab"],
        top_k=1,
        score_threshold=cfg.qa_lead_threshold,
    )

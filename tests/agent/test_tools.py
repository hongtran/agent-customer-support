import pytest
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock
from agent_customer_support.agent.tools import TOOL_DEFS, ToolContext, dispatch
from agent_customer_support.models import Flow, FlowStep, CustomerProfile
from agent_customer_support.agent import tools as tools_mod

pytestmark = pytest.mark.asyncio


def test_tool_defs_has_five_tools():
    names = {t["name"] for t in TOOL_DEFS}
    assert names == {
        "search_knowledge",
        "list_flows",
        "get_flow",
        "log_request",
        "escalate_to_human",
    }


async def test_dispatch_search_knowledge():
    rag = AsyncMock()
    rag.search.return_value = {"passages": ["p1"], "citations": ["c#1"], "top_confidence": 0.9}
    ctx = ToolContext(
        customer=CustomerProfile(customer_id="c1", name="C1", enabled_modules=["m"]),
        rag=rag,
        flow_store=AsyncMock(),
        backlog=AsyncMock(),
        escalator=AsyncMock(),
        conversation_id="cv1",
    )
    out = await dispatch("search_knowledge", {"query": "x"}, ctx)
    assert out["top_confidence"] == 0.9
    rag.search.assert_awaited_once()


async def test_dispatch_get_flow():
    fs = AsyncMock()
    fs.get.return_value = Flow(id="f1", title="t", module="m", steps=[FlowStep(id="s1", say="hi")])
    ctx = ToolContext(
        customer=CustomerProfile(customer_id="c1", name="C1"),
        rag=AsyncMock(),
        flow_store=fs,
        backlog=AsyncMock(),
        escalator=AsyncMock(),
        conversation_id="cv1",
    )
    out = await dispatch("get_flow", {"flow_id": "f1"}, ctx)
    assert out["flow"]["id"] == "f1"


async def test_dispatch_log_request():
    backlog = AsyncMock()
    backlog.add.return_value = type("R", (), {"id": "r1"})()
    ctx = ToolContext(
        customer=CustomerProfile(customer_id="c1", name="C1"),
        rag=AsyncMock(),
        flow_store=AsyncMock(),
        backlog=backlog,
        escalator=AsyncMock(),
        conversation_id="cv1",
    )
    out = await dispatch("log_request", {"type": "feature", "summary": "thêm cột"}, ctx)
    assert out["logged"] is True and out["request_id"] == "r1"
    backlog.add.assert_awaited_once()


async def test_dispatch_wraps_in_span(monkeypatch):
    handle = MagicMock()
    calls: dict = {}

    @contextmanager
    def fake_span(name, *, input=None, metadata=None):
        calls["name"] = name
        yield handle

    monkeypatch.setattr(tools_mod.tracing, "span", fake_span)
    rag = AsyncMock()
    rag.search.return_value = {"passages": ["p"], "citations": [], "top_confidence": 0.9}
    ctx = ToolContext(
        customer=CustomerProfile(customer_id="c1", name="C1", enabled_modules=["m"]),
        rag=rag,
        flow_store=AsyncMock(),
        backlog=AsyncMock(),
        escalator=AsyncMock(),
        conversation_id="cv1",
    )
    out = await dispatch("search_knowledge", {"query": "x"}, ctx)
    assert calls["name"] == "tool.search_knowledge"
    handle.update.assert_called_once()
    assert out["top_confidence"] == 0.9

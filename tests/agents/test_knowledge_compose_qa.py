import pytest
import agent_customer_support.agents.knowledge as kn
from agent_customer_support.agents.knowledge import KnowledgeAgent
from agent_customer_support.agents.prompts import (
    KNOWLEDGE_COMPOSE_PROMPT,
    KNOWLEDGE_COMPOSE_PROMPT_WITH_QA,
)
from agent_customer_support.config import get_settings

pytestmark = pytest.mark.asyncio


def _capture(monkeypatch):
    cap = {}

    def fake_complete_text(messages, system, model):
        cap["content"] = messages[0]["content"]
        cap["system_text"] = system[-1]["text"] if isinstance(system, list) else system
        return "Anh/Chị vui lòng làm theo hướng dẫn."

    monkeypatch.setattr(kn, "complete_text", fake_complete_text)
    return cap


async def test_no_qa_uses_two_source_prompt(monkeypatch):
    cap = _capture(monkeypatch)
    agent = KnowledgeAgent()
    await agent._compose("q", ["guide passage"], "", get_settings())
    assert cap["system_text"] == KNOWLEDGE_COMPOSE_PROMPT
    assert "ĐÁP ÁN CS" not in cap["content"]


async def test_qa_leads_uses_three_source_authoritative_block(monkeypatch):
    cap = _capture(monkeypatch)
    agent = KnowledgeAgent()
    await agent._compose(
        "q", ["guide"], "", get_settings(), qa_passages=["cs answer"], qa_leads=True
    )
    assert cap["system_text"] == KNOWLEDGE_COMPOSE_PROMPT_WITH_QA
    assert "ĐÁP ÁN CS XÁC NHẬN" in cap["content"]
    assert "ưu tiên cao nhất" in cap["content"]
    assert "cs answer" in cap["content"]


async def test_qa_supplementary_uses_three_source_supplementary_block(monkeypatch):
    cap = _capture(monkeypatch)
    agent = KnowledgeAgent()
    await agent._compose(
        "q", ["guide"], "", get_settings(), qa_passages=["cs answer"], qa_leads=False
    )
    assert cap["system_text"] == KNOWLEDGE_COMPOSE_PROMPT_WITH_QA
    assert "bổ trợ" in cap["content"]
    assert "ưu tiên cao nhất" not in cap["content"]

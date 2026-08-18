import pytest
from unittest.mock import patch
from agent_customer_support.agents.guardrail import GuardrailAgent
from agent_customer_support.llm.schemas import GuardrailVerdict

pytestmark = pytest.mark.asyncio


async def test_empty_input_blocked():
    g = GuardrailAgent()
    res = await g.check_input("   ")
    assert res["pass"] is False


async def test_oversized_input_blocked():
    g = GuardrailAgent()
    res = await g.check_input("x" * 6000)
    assert res["pass"] is False


async def test_normal_input_passes():
    g = GuardrailAgent()
    res = await g.check_input("làm sao tạo phiếu yêu cầu?")
    assert res["pass"] is True


async def test_output_flagged_by_llm():
    g = GuardrailAgent()
    with patch(
        "agent_customer_support.agents.guardrail.complete_structured",
        return_value=GuardrailVerdict(flag=True, reason="off-topic"),
    ):
        res = await g.check_output("nội dung ngoài phạm vi")
    assert res["pass"] is False
    assert res["reason"] == "off-topic"


async def test_output_passes_when_not_flagged():
    g = GuardrailAgent()
    with patch(
        "agent_customer_support.agents.guardrail.complete_structured",
        return_value=GuardrailVerdict(flag=False, reason=""),
    ):
        res = await g.check_output("câu trả lời hợp lệ")
    assert res["pass"] is True


async def test_output_fails_open_when_no_verdict():
    """A judge that could not answer must never silence an already-paid-for reply.
    Deliberately the opposite direction from triage's fail-safe."""
    g = GuardrailAgent()
    with patch("agent_customer_support.agents.guardrail.complete_structured", return_value=None):
        res = await g.check_output("câu trả lời hợp lệ")
    assert res["pass"] is True
    assert res["reason"] == ""


async def test_flagged_with_empty_reason_gets_placeholder():
    g = GuardrailAgent()
    with patch(
        "agent_customer_support.agents.guardrail.complete_structured",
        return_value=GuardrailVerdict(flag=True, reason=""),
    ):
        res = await g.check_output("nội dung đáng ngờ")
    assert res["pass"] is False
    assert res["reason"] == "flagged"

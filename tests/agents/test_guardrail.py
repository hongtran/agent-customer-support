import pytest
from unittest.mock import patch
from agent_customer_support.agents.guardrail import GuardrailAgent
from agent_customer_support.llm.schemas import GroundingVerdict

pytestmark = pytest.mark.asyncio

_SOURCES = ["Vào menu Phiếu yêu cầu, nhấn Tạo mới."]


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


async def test_ungrounded_reply_is_flagged():
    g = GuardrailAgent()
    with patch(
        "agent_customer_support.agents.guardrail.complete_structured",
        return_value=GroundingVerdict(
            grounded=False, reason="bịa nút", unsupported_claims=["nhấn nút Xuất Excel"]
        ),
    ):
        res = await g.check_output("Nhấn nút Xuất Excel ở góc phải.", _SOURCES)
    assert res["pass"] is False
    assert res["reason"] == "bịa nút"
    assert res["unsupported_claims"] == ["nhấn nút Xuất Excel"]


async def test_grounded_reply_passes():
    g = GuardrailAgent()
    with patch(
        "agent_customer_support.agents.guardrail.complete_structured",
        return_value=GroundingVerdict(grounded=True, reason="", unsupported_claims=[]),
    ):
        res = await g.check_output("Vào Phiếu yêu cầu rồi nhấn Tạo mới.", _SOURCES)
    assert res["pass"] is True


async def test_no_cited_passages_skips_the_judge():
    """Nothing to judge against, so no call at all.

    This is the common case, not an edge one: every non-knowledge route and every
    clarify/no-answer reply arrives with an empty list. Calling the judge there would
    flag correct replies for lacking sources they never claimed to have — and would
    spend a model call on every single turn.
    """
    g = GuardrailAgent()
    with patch("agent_customer_support.agents.guardrail.complete_structured") as llm:
        res = await g.check_output("Bạn đang muốn tạo loại phiếu nào?", [])
    assert res["pass"] is True
    llm.assert_not_called()


async def test_fails_open_when_no_verdict():
    """A judge that could not answer must never silence an already-paid-for reply.
    Deliberately the opposite direction from triage's fail-safe."""
    g = GuardrailAgent()
    with patch("agent_customer_support.agents.guardrail.complete_structured", return_value=None):
        res = await g.check_output("câu trả lời hợp lệ", _SOURCES)
    assert res["pass"] is True
    assert res["reason"] == ""


async def test_flagged_with_empty_reason_gets_placeholder():
    g = GuardrailAgent()
    with patch(
        "agent_customer_support.agents.guardrail.complete_structured",
        return_value=GroundingVerdict(grounded=False, reason="", unsupported_claims=[]),
    ):
        res = await g.check_output("nội dung đáng ngờ", _SOURCES)
    assert res["pass"] is False
    assert res["reason"] == "ungrounded"

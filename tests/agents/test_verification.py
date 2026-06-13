import pytest
from unittest.mock import patch, AsyncMock
from agent_customer_support.agents.verification import IssueVerificationAgent
from agent_customer_support.agents.context import TurnContext
from agent_customer_support.models import (
    CustomerProfile, SessionState, Conversation, Attachment,
)

pytestmark = pytest.mark.asyncio


def _ctx(message, attachments=None) -> TurnContext:
    s = SessionState(conversation_id="cv1", pending="verify_issue",
                     pending_context={"module": "m", "summary": "A bị lỗi"})
    return TurnContext(
        customer=CustomerProfile(customer_id="c1", name="C1"),
        session=s,
        conversation=Conversation(conversation_id="cv1", customer_id="c1"),
        message=message, attachments=attachments or [],
        transcript=f"user: {message}",
        rag=AsyncMock(), backlog=AsyncMock(), escalator=AsyncMock(),
    )


async def test_insufficient_evidence_asks_more():
    with patch("agent_customer_support.agents.verification.complete_with_tools",
               return_value={"stop_reason": "end_turn",
                             "text": "Bạn gửi giúp ảnh lỗi nhé?", "tool_calls": []}):
        res = await IssueVerificationAgent().run(_ctx("nó cứ lỗi thôi"))
    assert res.evidence_complete is False
    assert "ảnh" in res.reply


async def test_evidence_ready_marks_complete():
    att = Attachment(kind="image", media_type="image/png", data="QUJD")
    with patch("agent_customer_support.agents.verification.complete_with_tools",
               return_value={"stop_reason": "end_turn",
                             "text": "Đã nhận đủ thông tin. [[evidence_ready]]",
                             "tool_calls": []}):
        res = await IssueVerificationAgent().run(_ctx("đây là ảnh", [att]))
    assert res.evidence_complete is True
    assert "[[evidence_ready]]" not in res.reply

import pytest
from unittest.mock import patch, AsyncMock
from agent_customer_support.agents.verification import IssueVerificationAgent
from agent_customer_support.agents.context import TurnContext
from agent_customer_support.models import (
    CustomerProfile,
    SessionState,
    Conversation,
    Attachment,
)

pytestmark = pytest.mark.asyncio


def _ctx(message, attachments=None) -> TurnContext:
    s = SessionState(
        conversation_id="cv1",
        pending="verify_issue",
        pending_context={"module": "m", "summary": "A bị lỗi"},
    )
    return TurnContext(
        customer=CustomerProfile(customer_id="c1", name="C1"),
        session=s,
        conversation=Conversation(conversation_id="cv1", customer_id="c1"),
        message=message,
        attachments=attachments or [],
        transcript=f"user: {message}",
        rag=AsyncMock(),
        backlog=AsyncMock(),
        escalator=AsyncMock(),
    )


async def test_insufficient_evidence_asks_more():
    with patch(
        "agent_customer_support.agents.verification.complete_with_tools",
        return_value={
            "stop_reason": "end_turn",
            "text": "Bạn gửi giúp ảnh lỗi nhé?",
            "tool_calls": [],
        },
    ):
        res = await IssueVerificationAgent().run(_ctx("nó cứ lỗi thôi"))
    assert res.evidence_complete is False
    assert "ảnh" in res.reply


async def test_evidence_ready_marks_complete():
    att = Attachment(kind="image", media_type="image/png", data="QUJD")
    with patch(
        "agent_customer_support.agents.verification.complete_with_tools",
        return_value={
            "stop_reason": "end_turn",
            "text": "Đã nhận đủ thông tin. [[evidence_ready]]",
            "tool_calls": [],
        },
    ):
        res = await IssueVerificationAgent().run(_ctx("đây là ảnh", [att]))
    assert res.evidence_complete is True
    assert "[[evidence_ready]]" not in res.reply


async def test_verification_includes_prior_turns_in_messages():
    """Verification must send conversation history so the agent knows what bug
    is being verified, not just the bare follow-up message."""
    from agent_customer_support.models import Turn

    ctx = _ctx("đây là ảnh")
    ctx.conversation.turns = [
        Turn(role="user", content="thanh toán PYC không in được"),
        Turn(role="assistant", content="Bạn gửi ảnh chụp lỗi giúp mình nhé?"),
    ]
    captured: dict = {}

    def fake_complete(*, messages, tools, system, model=None):
        captured["messages"] = messages
        return {"stop_reason": "end_turn", "text": "Đã nhận. [[evidence_ready]]", "tool_calls": []}

    with patch(
        "agent_customer_support.agents.verification.complete_with_tools", side_effect=fake_complete
    ):
        await IssueVerificationAgent().run(ctx)

    # 2 prior turns + current turn = 3 messages
    assert len(captured["messages"]) == 3
    assert captured["messages"][0]["content"] == "thanh toán PYC không in được"
    assert captured["messages"][1]["role"] == "assistant"
    # current turn is last and may be a list (multimodal) or str
    last = captured["messages"][2]["content"]
    assert last == "đây là ảnh" or (isinstance(last, list) and last[0].get("text") == "đây là ảnh")

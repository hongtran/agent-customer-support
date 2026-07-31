import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_customer_support.agents.coordinator import Coordinator
from agent_customer_support.models import (
    AgentResult,
    Attachment,
    AttachmentRef,
    Conversation,
    CustomerProfile,
    SessionState,
    StoredAttachment,
)

pytestmark = pytest.mark.asyncio

IMG = Attachment(kind="image", media_type="image/png", data="QUJD")


def _coord():
    c = Coordinator()
    c.customers = AsyncMock()
    c.customers.get.return_value = CustomerProfile(customer_id="c1", name="C1")
    c.conversations = AsyncMock()
    c.conversations.load.return_value = Conversation(conversation_id="cv1", customer_id="c1")
    c.sessions = AsyncMock()
    c.sessions.get.return_value = SessionState(conversation_id="cv1")
    c.rag = AsyncMock()
    c.flow_store = AsyncMock()
    c.backlog = AsyncMock()
    c.escalator = AsyncMock()
    c.attachments = AsyncMock()
    c.guardrail = MagicMock()
    c.guardrail.check_input = AsyncMock(return_value={"pass": True, "reason": ""})
    c.guardrail.check_output = AsyncMock(return_value={"pass": True, "reason": ""})
    c.triage = MagicMock()
    c.knowledge = MagicMock()
    c.knowledge.run = AsyncMock(return_value=AgentResult(reply="Vào menu X.", resolved=True))
    c.flow = MagicMock()
    c.verification = MagicMock()
    c.escalation = MagicMock()
    c.sessions.get.return_value = SessionState(conversation_id="cv1", pending="knowledge_clarify")
    return c


def _persisted_user_turn(c):
    """The Turn passed to conversations.append for the user side."""
    return next(
        call.args[2]
        for call in c.conversations.append.await_args_list
        if call.args[2].role == "user"
    )


async def test_image_bytes_never_reach_the_conversation_record():
    c = _coord()
    c.attachments.put = AsyncMock(
        return_value=StoredAttachment(
            kind="image", media_type="image/png", s3_key="conversations/cv1/t/0.png", size_bytes=3
        )
    )
    c.attachments.presign = AsyncMock(
        return_value=AttachmentRef(kind="image", media_type="image/png", url="https://signed/0.png")
    )

    res = await c.handle_turn(
        customer_id="c1", conversation_id="cv1", message="cái này lỗi gì?", attachments=[IMG]
    )

    turn = _persisted_user_turn(c)
    assert turn.attachments[0].s3_key == "conversations/cv1/t/0.png"
    # the regression this whole change exists to prevent
    assert "QUJD" not in turn.model_dump_json()
    # and the UI gets a signed URL back for the turn it just sent
    assert [a.url for a in res.attachments] == ["https://signed/0.png"]


async def test_upload_is_keyed_by_the_persisted_turn_id():
    c = _coord()
    c.attachments.put = AsyncMock(
        return_value=StoredAttachment(
            kind="image", media_type="image/png", s3_key="k", size_bytes=3
        )
    )
    c.attachments.presign = AsyncMock(
        return_value=AttachmentRef(kind="image", media_type="image/png", url="u")
    )

    await c.handle_turn(customer_id="c1", conversation_id="cv1", message="x", attachments=[IMG])

    turn = _persisted_user_turn(c)
    conv_id, turn_id, index, _ = c.attachments.put.await_args.args
    assert (conv_id, turn_id, index) == ("cv1", turn.id, 0)


async def test_upload_failure_still_returns_the_reply():
    """The reply is already generated and paid for by the time we persist. An S3
    outage must cost the screenshot, not the answer."""
    c = _coord()
    c.attachments.put = AsyncMock(side_effect=RuntimeError("s3 down"))

    res = await c.handle_turn(
        customer_id="c1", conversation_id="cv1", message="cái này lỗi gì?", attachments=[IMG]
    )

    assert res.reply == "Vào menu X."
    assert res.attachments == []
    assert _persisted_user_turn(c).attachments == []


async def test_presign_failure_still_returns_the_reply():
    c = _coord()
    c.attachments.put = AsyncMock(
        return_value=StoredAttachment(
            kind="image", media_type="image/png", s3_key="k", size_bytes=3
        )
    )
    c.attachments.presign = AsyncMock(side_effect=RuntimeError("signing broke"))

    res = await c.handle_turn(
        customer_id="c1", conversation_id="cv1", message="x", attachments=[IMG]
    )

    assert res.reply == "Vào menu X."
    assert res.attachments == []
    # the object is still stored and referenced, only the URL is missing
    assert _persisted_user_turn(c).attachments[0].s3_key == "k"


async def test_no_attachments_skips_s3_entirely():
    c = _coord()
    c.attachments.put = AsyncMock()
    await c.handle_turn(customer_id="c1", conversation_id="cv1", message="x", attachments=[])
    c.attachments.put.assert_not_awaited()

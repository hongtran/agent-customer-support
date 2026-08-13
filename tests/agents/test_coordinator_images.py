from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_customer_support.agents.coordinator import Coordinator
from agent_customer_support.models import (
    AgentResult,
    Conversation,
    CustomerProfile,
    SessionState,
)

pytestmark = pytest.mark.asyncio

SLUG = "phong_thi_nghiem"
ICON_MARKER = f"[[img:icon:{SLUG}/image24.png]]"
SCREEN_MARKER = f"[[img:screen:{SLUG}/image23.png]]"


def _coord(reply: str):
    c = Coordinator()
    c.customers = AsyncMock()
    c.customers.get.return_value = CustomerProfile(customer_id="c1", name="C1")
    c.conversations = AsyncMock()
    c.conversations.load.return_value = Conversation(conversation_id="cv1", customer_id="c1")
    c.sessions = AsyncMock()
    # pending routes straight to knowledge, skipping triage
    c.sessions.get.return_value = SessionState(conversation_id="cv1", pending="knowledge_clarify")
    c.rag = AsyncMock()
    c.flow_store = AsyncMock()
    c.backlog = AsyncMock()
    c.escalator = AsyncMock()
    c.attachments = AsyncMock()
    c.doc_images = AsyncMock()
    c.doc_images.presign = AsyncMock(side_effect=lambda slug, name: f"https://signed/{slug}/{name}")
    c.guardrail = MagicMock()
    c.guardrail.check_input = AsyncMock(return_value={"pass": True, "reason": ""})
    c.guardrail.check_output = AsyncMock(return_value={"pass": True, "reason": ""})
    c.triage = MagicMock()
    c.knowledge = MagicMock()
    c.knowledge.run = AsyncMock(return_value=AgentResult(reply=reply, resolved=True))
    c.flow = MagicMock()
    c.verification = MagicMock()
    c.escalation = MagicMock()
    return c


def _persisted_assistant_turn(c):
    return next(
        call.args[2]
        for call in c.conversations.append.await_args_list
        if call.args[2].role == "assistant"
    )


async def _turn(c):
    return await c.handle_turn(
        customer_id="c1", conversation_id="cv1", message="cách tạo hồ sơ?", attachments=[]
    )


async def test_response_carries_urls_while_the_stored_turn_keeps_markers():
    """The whole point of the split: a presigned URL expires, so persisting one would
    archive a dead link and feed 500 chars of signature into next turn's transcript."""
    c = _coord(f"Anh/Chị nhấn {ICON_MARKER} để tạo hồ sơ.")
    res = await _turn(c)

    assert res.reply == (f"Anh/Chị nhấn ![icon](https://signed/{SLUG}/image24.png) để tạo hồ sơ.")
    stored = _persisted_assistant_turn(c).content
    assert stored == f"Anh/Chị nhấn {ICON_MARKER} để tạo hồ sơ."
    assert "X-Amz" not in stored and "https://" not in stored


async def test_kind_survives_into_the_alt_text():
    """It is the only channel the widget has for choosing inline glyph vs preview."""
    c = _coord(f"Màn hình xử lý:\n{SCREEN_MARKER}")
    res = await _turn(c)
    assert res.reply == f"Màn hình xử lý:\n![screen](https://signed/{SLUG}/image23.png)"


async def test_each_image_is_signed_once():
    c = _coord(f"{ICON_MARKER} rồi {ICON_MARKER} lại {SCREEN_MARKER}")
    await _turn(c)
    signed = {call.args for call in c.doc_images.presign.await_args_list}
    assert signed == {(SLUG, "image24.png"), (SLUG, "image23.png")}


async def test_presign_failure_drops_the_images_not_the_answer():
    c = _coord(f"Anh/Chị nhấn {ICON_MARKER} để tạo hồ sơ.")
    c.doc_images.presign = AsyncMock(side_effect=RuntimeError("s3 down"))

    res = await _turn(c)

    assert res.reply == "Anh/Chị nhấn để tạo hồ sơ."
    assert "img:" not in res.reply
    # the reply was still persisted with its markers, so a retry can resign them
    assert ICON_MARKER in _persisted_assistant_turn(c).content


async def test_a_marker_free_reply_makes_no_s3_call():
    c = _coord("Anh/Chị vui lòng vào menu Nguyên nhân.")
    res = await _turn(c)
    assert res.reply == "Anh/Chị vui lòng vào menu Nguyên nhân."
    c.doc_images.presign.assert_not_awaited()


async def test_the_output_guardrail_judges_prose_not_signatures():
    """Resolution runs after the guardrail, so a 500-char signed URL never reaches it."""
    c = _coord(f"Anh/Chị nhấn {ICON_MARKER} để tạo hồ sơ.")
    await _turn(c)
    checked = c.guardrail.check_output.await_args.args[0]
    assert ICON_MARKER in checked
    assert "https://" not in checked


async def test_a_flagged_reply_is_replaced_and_needs_no_signing():
    c = _coord(f"Anh/Chị nhấn {ICON_MARKER} để tạo hồ sơ.")
    c.guardrail.check_output = AsyncMock(return_value={"pass": False, "reason": "off-topic"})
    res = await _turn(c)
    assert "img:" not in res.reply
    c.doc_images.presign.assert_not_awaited()

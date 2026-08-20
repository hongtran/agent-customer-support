from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_customer_support.agents.context import TurnContext
from agent_customer_support.agents.knowledge import KnowledgeAgent
from agent_customer_support.models import Conversation, CustomerProfile, SessionState

pytestmark = pytest.mark.asyncio

SLUG = "phong_thi_nghiem"
SCREEN_MARKER = f"[[img:screen:{SLUG}/image23.png]]"
ICON_MARKER = f"[[img:icon:{SLUG}/image24.png]]"

# A passage shaped like the real thing: a standalone screenshot and a table row whose
# first cell is a button glyph.
PASSAGE = (
    "## 4.1. Tạo CV KPH\n![](media/image23.png)\n| ![](media/image24.png) | Nhấn để tạo hồ sơ. |"
)


def _ctx(doc_images_store=None, message="cách tạo hồ sơ?") -> TurnContext:
    return TurnContext(
        customer=CustomerProfile(customer_id="c1", name="C1"),
        session=SessionState(conversation_id="cv1", selected_applications=["Phòng thí nghiệm"]),
        conversation=Conversation(conversation_id="cv1", customer_id="c1"),
        message=message,
        transcript=f"user: {message}",
        rag=AsyncMock(),
        doc_images=doc_images_store,
        backlog=AsyncMock(),
        flow_store=AsyncMock(),
        qa_store=AsyncMock(),
    )


def _store(names):
    """A DocImageStore stand-in whose catalog reports `names` for SLUG."""
    store = MagicMock()
    store.catalog = AsyncMock(return_value={SLUG: set(names)})
    return store


def _search_result(passage=PASSAGE, application=SLUG):
    return {
        "passages": [passage],
        "citations": ["doc1"],
        "metas": [{"application": application, "confidence": 0.9}],
        "top_confidence": 0.9,
    }


async def _run(ctx, composed, search=None):
    """Run KnowledgeAgent with retrieval and composition both stubbed."""
    ctx.rag.search_with_fallback = AsyncMock(
        return_value=search if search is not None else _search_result()
    )
    ctx.rag.search = AsyncMock(  # qa collection
        return_value={"passages": [], "citations": [], "top_confidence": 0.0}
    )
    with patch(
        "agent_customer_support.agents.knowledge.complete_text",
        return_value=composed,
    ) as llm:
        res = await KnowledgeAgent().run(ctx)
    return res, llm


def _compose_payload(llm) -> str:
    """The user content of the compose call (the last LLM call in the run)."""
    return llm.call_args_list[-1].kwargs["messages"][0]["content"]


async def test_available_refs_reach_the_composer_as_scoped_markers():
    ctx = _ctx(_store({"image23.png", "image24.png"}))
    _, llm = await _run(ctx, "Anh/Chị vui lòng mở màn hình.")

    payload = _compose_payload(llm)
    assert SCREEN_MARKER in payload
    assert ICON_MARKER in payload
    # invariant 3: no raw ref is ever handed to the model
    assert "media/image" not in payload


async def test_marker_the_composer_used_survives_into_the_reply():
    ctx = _ctx(_store({"image23.png", "image24.png"}))
    res, _ = await _run(ctx, f"Anh/Chị vui lòng nhấn {ICON_MARKER} để tạo hồ sơ.")
    assert res.reply == f"Anh/Chị vui lòng nhấn {ICON_MARKER} để tạo hồ sơ."


async def test_document_with_no_images_in_the_bucket_yields_a_plain_text_answer():
    """The headline requirement. Same code path as a document that has images — the
    empty catalog is the only difference."""
    ctx = _ctx(_store(set()))
    composed = "Anh/Chị vui lòng chọn Phiếu yêu cầu thử nghiệm."
    res, llm = await _run(ctx, composed)

    assert res.reply == composed
    payload = _compose_payload(llm)
    assert "img:" not in payload and "media/image" not in payload


async def test_chunk_without_an_application_offers_no_images():
    """A global document is visible to every customer but has no prefix to resolve
    against, so its refs are dropped rather than guessed at."""
    ctx = _ctx(_store({"image23.png"}))
    _, llm = await _run(
        ctx, "Anh/Chị vui lòng mở màn hình.", search=_search_result(application=None)
    )
    payload = _compose_payload(llm)
    assert "img:" not in payload and "media/image" not in payload


async def test_invented_marker_never_reaches_the_persisted_reply():
    ctx = _ctx(_store({"image23.png", "image24.png"}))
    res, _ = await _run(ctx, f"Xem hình. [[img:screen:{SLUG}/image999.png]] Rồi bấm Lưu.")
    assert "img:" not in res.reply
    assert "Xem hình." in res.reply and "Rồi bấm Lưu." in res.reply


async def test_reply_is_capped_at_the_configured_maximum():
    names = {f"image{n}.png" for n in range(1, 12)}
    ctx = _ctx(_store(names))
    composed = " ".join(f"[[img:icon:{SLUG}/image{n}.png]]" for n in range(1, 12))
    res, _ = await _run(ctx, composed)
    assert res.reply.count("[[img:") == 5


async def test_images_are_offered_on_the_suspected_bug_path_too():
    ctx = _ctx(_store({"image23.png"}))
    res, _ = await _run(
        ctx, f"Chức năng này đáng lẽ chạy. {SCREEN_MARKER} [[suspected_bug:phong_thi_nghiem]]"
    )
    assert res.suspected_bug is True
    assert SCREEN_MARKER in res.reply
    assert "[[suspected_bug" not in res.reply


async def test_no_answer_is_still_detected_when_a_marker_inflates_the_reply():
    """parse_markers treats a long reply plus [[no_answer]] as a spurious hedge. Image
    markers are ~40 chars each, so counting them could suppress a real miss."""
    ctx = _ctx(_store({"image23.png"}))
    res, _ = await _run(ctx, f"Chưa rõ. {SCREEN_MARKER} [[no_answer]]")
    # first miss asks a clarifying question rather than answering
    assert res.resolved is None
    assert ctx.session.pending == "knowledge_clarify"


async def test_no_store_handle_degrades_to_text_only():
    ctx = _ctx(None)
    _, llm = await _run(ctx, "Anh/Chị vui lòng mở màn hình.")
    payload = _compose_payload(llm)
    assert "img:" not in payload and "media/image" not in payload


async def test_availability_is_not_looked_up_for_documents_without_refs():
    store = _store({"image23.png"})
    ctx = _ctx(store)
    await _run(ctx, "Xong.", search=_search_result(passage="Không có hình ở đây."))
    store.catalog.assert_not_awaited()

import pytest
from contextlib import contextmanager
from unittest.mock import MagicMock

from qdrant_client import AsyncQdrantClient, models

from agent_customer_support.rag_client import RagClient
import agent_customer_support.rag_client as rag_client_mod

pytestmark = pytest.mark.asyncio

COLLECTION = "cenlab"


def _point(pid, vector, *, source_doc_id, doc_type="guide", application="Lab", text="noi dung"):
    return models.PointStruct(
        id=pid,
        vector=vector,
        payload={
            "page_content": text,
            "metadata": {
                "source_doc_id": source_doc_id,
                "doc_type": doc_type,
                "application": application,
            },
        },
    )


async def _client_with(points):
    # search() resolves the logical name to the physical `_v3` collection, so the
    # in-memory store must be created under the resolved name.
    physical = rag_client_mod._normalize_collection(COLLECTION)
    client = AsyncQdrantClient(location=":memory:")
    await client.create_collection(
        physical,
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    await client.upsert(physical, points=points)
    return client


def test_normalize_collection_adds_v3_suffix():
    assert rag_client_mod._normalize_collection("cenlab") == "cenlab_v3"
    # _v2 is replaced, not appended to
    assert rag_client_mod._normalize_collection("cenlab_v2") == "cenlab_v3"
    # already-versioned names pass through unchanged (no double suffix)
    assert rag_client_mod._normalize_collection("cenlab_v3") == "cenlab_v3"


@pytest.fixture(autouse=True)
def _stub_embed(monkeypatch):
    async def fake_embed(text):
        return [1.0, 0.0]

    monkeypatch.setattr(rag_client_mod, "embed_query", fake_embed)


async def test_threshold_keeps_only_scores_at_or_above_default():
    # q=[1,0]; A->cos 1.0, C->cos 0.6, D->cos 0.3 (below 0.6 default)
    client = await _client_with(
        [
            _point(1, [1.0, 0.0], source_doc_id="hdsd#1", text="Buoc 1"),
            _point(2, [0.6, 0.8], source_doc_id="hdsd#2", text="Buoc 2"),
            _point(3, [0.3, 0.9539392], source_doc_id="hdsd#3", text="Buoc 3"),
        ]
    )
    res = await RagClient(client=client).search("q", collection=COLLECTION)
    assert res["top_confidence"] == 1.0
    assert set(res["citations"]) == {"hdsd#1", "hdsd#2"}  # hdsd#3 below 0.6


async def test_floor_relax_when_nothing_clears_threshold():
    client = await _client_with(
        [
            _point(1, [0.6, 0.8], source_doc_id="hdsd#2"),
            _point(2, [0.3, 0.9539392], source_doc_id="hdsd#3"),
        ]
    )
    # threshold 0.9 clears nothing -> relax to 0.25 floor -> both (>=0.25) returned
    res = await RagClient(client=client).search("q", collection=COLLECTION, score_threshold=0.9)
    assert set(res["citations"]) == {"hdsd#2", "hdsd#3"}


async def test_application_filter_then_silent_relax():
    client = await _client_with(
        [
            _point(1, [1.0, 0.0], source_doc_id="hdsd#1", application="Lab"),
            _point(2, [0.6, 0.8], source_doc_id="hdsd#2", application="Other"),
        ]
    )
    res = await RagClient(client=client).search("q", collection=COLLECTION, applications=["Lab"])
    assert set(res["citations"]) == {"hdsd#1"}

    # filter that matches nothing is silently relaxed to unfiltered
    res2 = await RagClient(client=client).search(
        "q", collection=COLLECTION, applications=["DoesNotExist"]
    )
    assert set(res2["citations"]) == {"hdsd#1", "hdsd#2"}


async def test_dedup_keeps_highest_scoring_chunk_per_source():
    client = await _client_with(
        [
            _point(1, [1.0, 0.0], source_doc_id="hdsd#1", text="best"),
            _point(2, [0.6, 0.8], source_doc_id="hdsd#1", text="worse"),
        ]
    )
    res = await RagClient(client=client).search("q", collection=COLLECTION)
    assert res["citations"] == ["hdsd#1"]
    assert res["passages"] == ["best"]
    assert res["top_confidence"] == 1.0


async def test_grounding_note_is_neutral_hint():
    client = await _client_with(
        [
            _point(1, [1.0, 0.0], source_doc_id="hdsd#1"),
        ]
    )
    res = await RagClient(client=client).search("q", collection=COLLECTION)
    note = res["grounding_note"]
    assert "log_request" not in note
    assert "clarification" not in note
    assert "confidence" in note.lower()


async def test_search_invokes_tracing_span(monkeypatch):
    handle = MagicMock()
    calls: dict = {}

    @contextmanager
    def fake_span(name, *, input=None, metadata=None):
        calls["name"] = name
        yield handle

    monkeypatch.setattr(rag_client_mod.tracing, "span", fake_span)

    client = await _client_with(
        [
            _point(1, [1.0, 0.0], source_doc_id="hdsd#5"),
        ]
    )
    res = await RagClient(client=client).search("q", collection=COLLECTION)

    assert calls["name"] == "rag.search"
    handle.update.assert_called_once()
    assert res["top_confidence"] == 1.0

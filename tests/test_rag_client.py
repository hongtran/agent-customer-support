import pytest
from contextlib import contextmanager
from unittest.mock import MagicMock

from qdrant_client import AsyncQdrantClient, models

from agent_customer_support.rag_client import RagClient
import agent_customer_support.rag_client as rag_client_mod

pytestmark = pytest.mark.asyncio

COLLECTION = "cenlab"


_OMIT = object()


def _point(pid, vector, *, source_doc_id, doc_type="guide", application="Lab", text="noi dung"):
    """Build a chunk point. Pass application=_OMIT to leave the key out of the
    payload entirely (vs. application=None for an explicit null) — the global-document
    filter has to handle both shapes."""
    metadata = {"source_doc_id": source_doc_id, "doc_type": doc_type}
    if application is not _OMIT:
        metadata["application"] = application
    return models.PointStruct(
        id=pid,
        vector=vector,
        payload={"page_content": text, "metadata": metadata},
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


async def test_application_filter_is_hard():
    client = await _client_with(
        [
            _point(1, [1.0, 0.0], source_doc_id="hdsd#1", application="Lab"),
            _point(2, [0.6, 0.8], source_doc_id="hdsd#2", application="Other"),
        ]
    )
    res = await RagClient(client=client).search("q", collection=COLLECTION, applications=["Lab"])
    assert set(res["citations"]) == {"hdsd#1"}

    # A filter matching nothing returns nothing — it is NOT relaxed to unfiltered.
    # Another application's docs must never answer a scoped question.
    res2 = await RagClient(client=client).search(
        "q", collection=COLLECTION, applications=["DoesNotExist"]
    )
    assert res2["citations"] == []
    assert res2["passages"] == []
    assert res2["top_confidence"] == 0.0


async def test_display_name_from_ui_matches_slug_in_payload():
    """Regression: the widget sends display names, Qdrant stores slugs.

    Filtering on the raw display name matched zero documents — with a hard filter
    that means every scoped question answered "no idea". search() must translate.
    """
    client = await _client_with(
        [
            _point(1, [1.0, 0.0], source_doc_id="hdsd#1", application="lay_mau_quan_trac"),
            _point(2, [0.9, 0.4359], source_doc_id="hdsd#2", application="quan_ly_kho"),
        ]
    )
    res = await RagClient(client=client).search(
        "q", collection=COLLECTION, applications=["Lấy mẫu - Quan trắc"]
    )
    assert set(res["citations"]) == {"hdsd#1"}

    # and the slug form keeps working, so existing callers are unaffected
    res2 = await RagClient(client=client).search(
        "q", collection=COLLECTION, applications=["lay_mau_quan_trac"]
    )
    assert set(res2["citations"]) == {"hdsd#1"}


async def test_untagged_documents_are_global():
    client = await _client_with(
        [
            _point(1, [1.0, 0.0], source_doc_id="hdsd#null", application=None),
            _point(2, [1.0, 0.0], source_doc_id="hdsd#missing", application=_OMIT),
            _point(3, [1.0, 0.0], source_doc_id="hdsd#other", application="Other"),
        ]
    )
    res = await RagClient(client=client).search("q", collection=COLLECTION, applications=["Lab"])
    # explicit null and a missing key both count as global; "Other" is still excluded
    assert set(res["citations"]) == {"hdsd#null", "hdsd#missing"}


async def test_doc_type_filter():
    client = await _client_with(
        [
            _point(1, [1.0, 0.0], source_doc_id="hdsd#1", doc_type="guide"),
            _point(2, [1.0, 0.0], source_doc_id="qa#1", doc_type="qa"),
        ]
    )
    res = await RagClient(client=client).search("q", collection=COLLECTION, doc_type="qa")
    assert set(res["citations"]) == {"qa#1"}


async def test_nothing_clears_threshold_returns_empty():
    client = await _client_with(
        [
            _point(1, [0.6, 0.8], source_doc_id="hdsd#2"),
            _point(2, [0.3, 0.9539392], source_doc_id="hdsd#3"),
        ]
    )
    res = await RagClient(client=client).search("q", collection=COLLECTION, score_threshold=0.9)
    assert res["citations"] == []
    assert res["top_confidence"] == 0.0


async def test_global_search_caps_chunks_per_source_at_per_doc():
    # Global product search (no application scope) keeps at most per_doc=2 chunks of
    # one source document, dropping the third so a single guide can't crowd out others.
    client = await _client_with(
        [
            _point(1, [1.0, 0.0], source_doc_id="hdsd#1", text="best"),
            _point(2, [0.8, 0.6], source_doc_id="hdsd#1", text="mid"),
            _point(3, [0.6, 0.8], source_doc_id="hdsd#1", text="worst"),
        ]
    )
    res = await RagClient(client=client).search("q", collection=COLLECTION)
    assert res["citations"] == ["hdsd#1"]
    assert res["passages"] == ["best", "mid"]  # "worst" dropped by the per-doc cap
    assert res["top_confidence"] == 1.0


async def test_custom_per_doc_cap_is_respected():
    client = await _client_with(
        [
            _point(1, [1.0, 0.0], source_doc_id="hdsd#1", text="best"),
            _point(2, [0.8, 0.6], source_doc_id="hdsd#1", text="mid"),
        ]
    )
    res = await RagClient(client=client).search("q", collection=COLLECTION, per_doc=1)
    assert res["passages"] == ["best"]  # cap of 1 keeps only the top chunk per doc


async def test_specific_application_skips_dedup():
    # A scoped search is effectively one guide, so all its chunks are kept as ranked
    # (per_doc is ignored) rather than collapsing to a single passage.
    client = await _client_with(
        [
            _point(1, [1.0, 0.0], source_doc_id="hdsd#1", application="Lab", text="best"),
            _point(2, [0.8, 0.6], source_doc_id="hdsd#1", application="Lab", text="mid"),
            _point(3, [0.6, 0.8], source_doc_id="hdsd#1", application="Lab", text="worst"),
        ]
    )
    res = await RagClient(client=client).search("q", collection=COLLECTION, applications=["Lab"])
    assert res["citations"] == ["hdsd#1"]
    assert res["passages"] == ["best", "mid", "worst"]


async def test_per_doc_none_skips_dedup_on_global_search():
    # QA-style call: global search that opts out of collapsing so distinct records
    # (and repeated chunks) are all kept as ranked.
    client = await _client_with(
        [
            _point(1, [1.0, 0.0], source_doc_id="qa#1", text="best"),
            _point(2, [0.8, 0.6], source_doc_id="qa#1", text="mid"),
            _point(3, [0.6, 0.8], source_doc_id="qa#1", text="worst"),
        ]
    )
    res = await RagClient(client=client).search("q", collection=COLLECTION, per_doc=None)
    assert res["passages"] == ["best", "mid", "worst"]


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
    def fake_span(name, *, as_type="span", input=None, metadata=None):
        calls["name"] = name
        calls["as_type"] = as_type
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


# ---- widened retry when the selected application has nothing ----
#
# The tests above pin the hard-filter contract: a wrong application returns nothing and
# is never silently relaxed. search_with_fallback is the explicit, opt-in way out of
# that, for the case where the user simply picked the wrong module in the widget.


def _counting_client(client):
    """Wrap query_points so a test can count Qdrant round-trips."""
    calls = []
    inner = client.query_points

    async def counted(*args, **kwargs):
        calls.append(kwargs.get("query_filter"))
        return await inner(*args, **kwargs)

    client.query_points = counted
    return calls


async def _two_module_client():
    return await _client_with(
        [
            _point(1, [1.0, 0.0], source_doc_id="lab#1", application="phong_thi_nghiem"),
            _point(2, [1.0, 0.0], source_doc_id="qt#1", application="lay_mau_quan_trac"),
        ]
    )


async def test_wrong_application_falls_back_to_the_customers_other_modules():
    """The whole point: the answer lives in a module the user did not select."""
    client = await _two_module_client()
    res = await RagClient(client=client).search_with_fallback(
        "q",
        collection=COLLECTION,
        applications=["Đo lường - Hiệu chuẩn"],  # nothing indexed under this one
        fallback_applications=["Đo lường - Hiệu chuẩn", "Phòng thí nghiệm"],
    )
    assert res["citations"] == ["lab#1"]
    assert res["fallback_used"] is True
    # Display names in, slugs out — the scope actually filtered on.
    assert res["applications_used"] == ["do_luong_hieu_chuan", "phong_thi_nghiem"]


async def test_a_scope_that_answers_is_never_widened():
    """No retry, and no second round-trip, when the selection was right."""
    client = await _two_module_client()
    calls = _counting_client(client)
    res = await RagClient(client=client).search_with_fallback(
        "q",
        collection=COLLECTION,
        applications=["Phòng thí nghiệm"],
        fallback_applications=["Phòng thí nghiệm", "Lấy mẫu - Quan trắc"],
    )
    assert res["citations"] == ["lab#1"]
    assert res["fallback_used"] is False
    assert res["applications_used"] == ["phong_thi_nghiem"]
    assert len(calls) == 1


@pytest.mark.parametrize(
    "applications,fallback",
    [
        # Same set: the retry would repeat the query it just ran.
        (["Đo lường - Hiệu chuẩn"], ["Đo lường - Hiệu chuẩn"]),
        # Subset: can only reach documents the first query already covered.
        (["Đo lường - Hiệu chuẩn", "Mua sắm"], ["Mua sắm"]),
        # Nothing to widen to.
        (["Đo lường - Hiệu chuẩn"], None),
        (["Đo lường - Hiệu chuẩn"], []),
        # Already global — no scope is wider than no scope.
        (None, ["Phòng thí nghiệm"]),
    ],
)
async def test_a_retry_that_cannot_change_the_outcome_is_skipped(applications, fallback):
    client = await _two_module_client()
    calls = _counting_client(client)
    res = await RagClient(client=client).search_with_fallback(
        "q", collection=COLLECTION, applications=applications, fallback_applications=fallback
    )
    assert len(calls) == 1, "spent a Qdrant round-trip that could not change the answer"
    assert res["fallback_used"] is False


async def test_the_query_is_embedded_once_across_both_attempts(monkeypatch):
    """Why `search` and `search_with_fallback` share a `_query` body: re-embedding a
    byte-identical string on the retry would be a second paid Google call."""
    embeds = []

    async def counting_embed(text):
        embeds.append(text)
        return [1.0, 0.0]

    monkeypatch.setattr(rag_client_mod, "embed_query", counting_embed)
    client = await _two_module_client()
    calls = _counting_client(client)
    res = await RagClient(client=client).search_with_fallback(
        "q",
        collection=COLLECTION,
        applications=["Đo lường - Hiệu chuẩn"],
        fallback_applications=["Đo lường - Hiệu chuẩn", "Phòng thí nghiệm"],
    )
    assert res["fallback_used"] is True
    assert len(calls) == 2  # two searches...
    assert embeds == ["q"]  # ...one embedding


async def test_search_reports_the_scope_it_filtered_on():
    """`applications_used` is on every result, not just the fallback path."""
    client = await _two_module_client()
    plain = await RagClient(client=client).search(
        "q", collection=COLLECTION, applications=["Phòng thí nghiệm"]
    )
    assert plain["applications_used"] == ["phong_thi_nghiem"]
    assert plain["fallback_used"] is False

    unscoped = await RagClient(client=client).search("q", collection=COLLECTION)
    assert unscoped["applications_used"] is None


async def test_multi_application_scope_collapses_per_document():
    """A single-application scope is effectively one guide, so it keeps every chunk.
    A multi-application scope spans several guides — the situation per_doc exists for,
    and what a widened fallback always looks like."""
    points = [
        _point(i, [1.0, 0.0], source_doc_id="lab#1", application="phong_thi_nghiem", text=f"c{i}")
        for i in range(4)
    ]
    client = await _client_with(points)

    one = await RagClient(client=client).search(
        "q", collection=COLLECTION, applications=["Phòng thí nghiệm"], per_doc=2
    )
    assert len(one["passages"]) == 4  # no collapsing

    many = await RagClient(client=client).search(
        "q",
        collection=COLLECTION,
        applications=["Phòng thí nghiệm", "Lấy mẫu - Quan trắc"],
        per_doc=2,
    )
    assert len(many["passages"]) == 2  # one guide can't crowd out the rest

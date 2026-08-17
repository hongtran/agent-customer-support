import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from agent_customer_support.stores.doc_image_store import DocImageStore

pytestmark = pytest.mark.asyncio

SLUG = "phong_thi_nghiem"


class _FakePaginator:
    """Mimics aioboto3's async paginator: `paginate(...)` is sync, the result is an
    async iterator of pages."""

    def __init__(self, pages, on_call):
        self._pages = pages
        self._on_call = on_call

    def paginate(self, **kwargs):
        self._on_call(kwargs)
        pages = self._pages

        async def gen():
            for page in pages:
                yield page

        return gen()


def _client(pages, calls, error=None):
    """Patch get_client with a fake S3 whose listing returns `pages`, recording the
    kwargs of every paginate call in `calls`."""
    s3 = MagicMock()
    if error:
        s3.get_paginator.side_effect = error
    else:
        s3.get_paginator.return_value = _FakePaginator(pages, calls.append)

    @contextlib.asynccontextmanager
    async def fake_get_client():
        yield s3

    return patch(
        "agent_customer_support.stores.doc_image_store.get_client",
        fake_get_client,
    )


def _page(*keys):
    return {"Contents": [{"Key": k} for k in keys]}


async def test_names_lists_the_slug_prefix_and_returns_bare_file_names():
    calls: list[dict] = []
    with _client([_page(f"doc_images/{SLUG}/image1.png", f"doc_images/{SLUG}/image2.png")], calls):
        store = DocImageStore()
        assert await store.names(SLUG) == {"image1.png", "image2.png"}
    assert calls[0]["Prefix"] == f"doc_images/{SLUG}/"


async def test_names_is_cached_so_a_second_turn_makes_no_s3_call():
    calls: list[dict] = []
    with _client([_page(f"doc_images/{SLUG}/image1.png")], calls):
        store = DocImageStore()
        await store.names(SLUG)
        await store.names(SLUG)
    assert len(calls) == 1


async def test_an_empty_result_is_cached_too():
    """A document whose media was never uploaded must cost one listing per TTL, not one
    per turn — otherwise the common case is the expensive one."""
    calls: list[dict] = []
    with _client([{}], calls):
        store = DocImageStore()
        assert await store.names(SLUG) == set()
        assert await store.names(SLUG) == set()
    assert len(calls) == 1


async def test_expired_cache_is_relisted():
    calls: list[dict] = []
    with _client([_page(f"doc_images/{SLUG}/image1.png")], calls):
        store = DocImageStore()
        await store.names(SLUG)
        # Force expiry rather than sleeping through the real 600s TTL.
        store._cache[SLUG] = (0.0, {"stale.png"})
        assert await store.names(SLUG) == {"image1.png"}
    assert len(calls) == 2


async def test_listing_failure_degrades_to_no_images():
    """Losing a picture is the cheap failure; raising here would lose an answer that
    retrieval and composition have already been paid for."""
    err = ClientError({"Error": {"Code": "AccessDenied"}}, "ListObjectsV2")
    with _client([], [], error=err):
        store = DocImageStore()
        assert await store.names(SLUG) == set()


async def test_names_short_circuits_on_an_empty_slug():
    calls: list[dict] = []
    with _client([_page("x")], calls):
        store = DocImageStore()
        assert await store.names("") == set()
    assert calls == []


async def test_nested_keys_and_directory_markers_are_ignored():
    calls: list[dict] = []
    pages = [
        _page(
            f"doc_images/{SLUG}/",  # directory marker some tools create
            f"doc_images/{SLUG}/image1.png",
            f"doc_images/{SLUG}/nested/image2.png",  # deeper than the flat layout
        )
    ]
    with _client(pages, calls):
        store = DocImageStore()
        assert await store.names(SLUG) == {"image1.png"}


async def test_catalog_gathers_several_applications():
    calls: list[dict] = []
    with _client([_page(f"doc_images/{SLUG}/image1.png")], calls):
        store = DocImageStore()
        got = await store.catalog([SLUG, "quan_ly_kho"])
    assert set(got) == {SLUG, "quan_ly_kho"}
    assert len(calls) == 2


async def test_presign_signs_the_derived_key():
    s3 = MagicMock()
    s3.generate_presigned_url = AsyncMock(return_value="https://signed/x")

    @contextlib.asynccontextmanager
    async def fake_get_client():
        yield s3

    with patch(
        "agent_customer_support.stores.doc_image_store.get_client",
        fake_get_client,
    ):
        store = DocImageStore()
        assert await store.presign(SLUG, "image23.png") == "https://signed/x"

    params = s3.generate_presigned_url.await_args.kwargs["Params"]
    assert params["Key"] == f"doc_images/{SLUG}/image23.png"

import logging
import time
from typing import Iterable

from botocore.exceptions import ClientError

from agent_customer_support import doc_images
from agent_customer_support.config import get_settings
from agent_customer_support.stores.attachment_store import get_client

logger = logging.getLogger(__name__)


class DocImageStore:
    """Read-only catalog of the images extracted from the source user guides.

    Laid out as `<doc_images_prefix>/<application_slug>/imageNN.png`, so a key is derived
    straight from a chunk's `metadata.application` with no slug→prefix map to keep in
    sync. Uploaded by `scripts/upload_doc_images.py`; nothing in the request path writes
    here.

    `names` is what makes the feature self-limiting: whether a reply can show images is
    decided by what is in the bucket, so a document whose media has not been uploaded
    yields an empty set and a plain-text answer. That is the same code path a document
    *with* images takes — there is no per-document flag anywhere.

    Reuses attachment_store.get_client so both S3 callers share the LocalStack endpoint
    override and region.
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self.bucket = cfg.s3_bucket_doc_images
        # slug -> (expires_at, names). Empty results are cached too: a document with no
        # images should cost one listing per TTL, not one per turn.
        self._cache: dict[str, tuple[float, set[str]]] = {}

    async def names(self, slug: str) -> set[str]:
        """File names available for one application. Never raises.

        A listing failure degrades to "this document has no images", which costs a
        picture. Raising would cost the whole answer, and by the time the request path
        reaches here retrieval has already been paid for.
        """
        if not slug:
            return set()
        cfg = get_settings()
        hit = self._cache.get(slug)
        now = time.monotonic()
        if hit and hit[0] > now:
            return hit[1]

        prefix = f"{cfg.doc_images_prefix}/{slug}/"
        found: set[str] = set()
        try:
            async with get_client() as s3:
                paginator = s3.get_paginator("list_objects_v2")
                async for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                    for obj in page.get("Contents", []):
                        name = obj["Key"].removeprefix(prefix)
                        # Skip the directory-marker object some tools create, and anything
                        # nested deeper than the flat layout this store defines.
                        if name and "/" not in name:
                            found.add(name)
        except ClientError as exc:
            logger.warning("doc image listing failed for %s: %s", slug, exc)
            return set()

        self._cache[slug] = (now + cfg.doc_images_cache_seconds, found)
        return found

    async def catalog(self, slugs: Iterable[str]) -> dict[str, set[str]]:
        """Availability for several applications at once, in the shape
        `doc_images.rewrite_passages` wants. Sequential on purpose: a turn retrieves from
        one or two documents, and the per-slug cache means most turns make no call."""
        return {slug: await self.names(slug) for slug in slugs}

    async def presign(self, slug: str, name: str) -> str:
        """Short-lived GET URL. Signing is local — no network round trip — so this is
        cheap enough to do per image on every response."""
        cfg = get_settings()
        async with get_client() as s3:
            return await s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": doc_images.s3_key(slug, name)},
                ExpiresIn=cfg.s3_presign_expiry_seconds,
            )

import base64
import os
import uuid

import httpx
import pytest

from agent_customer_support.models import Attachment
from agent_customer_support.stores.attachment_store import AttachmentStore

pytestmark = pytest.mark.asyncio

# The DynamoDB item ceiling that base64 attachments used to breach. Anything at or
# above this size is exactly what the old code could not persist.
OLD_DYNAMO_CEILING = 300 * 1024


def _attachment(raw: bytes, media_type: str = "image/png") -> Attachment:
    return Attachment(kind="image", media_type=media_type, data=base64.b64encode(raw).decode())


async def test_put_stores_bytes_and_returns_key():
    st = AttachmentStore()
    await st.init()
    raw = os.urandom(2048)
    stored = await st.put(f"conv-{uuid.uuid4()}", "turn-1", 0, _attachment(raw))

    assert stored.s3_key.endswith("/0.png")
    assert stored.size_bytes == len(raw)
    # the reference carries no bytes — that is the whole point
    assert not hasattr(stored, "data")


async def test_presigned_url_returns_the_original_bytes():
    st = AttachmentStore()
    await st.init()
    raw = os.urandom(4096)
    stored = await st.put(f"conv-{uuid.uuid4()}", "turn-1", 0, _attachment(raw))
    ref = await st.presign(stored)

    async with httpx.AsyncClient() as c:
        res = await c.get(ref.url)
    assert res.status_code == 200
    assert res.content == raw
    assert ref.media_type == "image/png"


async def test_stores_image_far_larger_than_the_old_dynamo_limit():
    """Regression: a 1 MB screenshot used to fail the whole request with
    ValidationException (item size exceeded). It must now round-trip."""
    st = AttachmentStore()
    await st.init()
    raw = os.urandom(1024 * 1024)
    assert len(raw) > OLD_DYNAMO_CEILING

    stored = await st.put(f"conv-{uuid.uuid4()}", "turn-1", 0, _attachment(raw))
    assert stored.size_bytes == len(raw)

    async with httpx.AsyncClient() as c:
        res = await c.get((await st.presign(stored)).url)
    assert res.status_code == 200 and len(res.content) == len(raw)


async def test_key_is_prefixed_by_conversation():
    # lifecycle rules and per-customer purges rely on this prefix layout
    st = AttachmentStore()
    cid = f"conv-{uuid.uuid4()}"
    stored = await st.put(cid, "turn-9", 2, _attachment(b"x" * 16, "image/jpeg"))
    assert stored.s3_key == f"conversations/{cid}/turn-9/2.jpg"


async def test_multiple_attachments_get_distinct_keys():
    st = AttachmentStore()
    cid = f"conv-{uuid.uuid4()}"
    a = await st.put(cid, "turn-1", 0, _attachment(b"a" * 32))
    b = await st.put(cid, "turn-1", 1, _attachment(b"b" * 32))
    assert a.s3_key != b.s3_key

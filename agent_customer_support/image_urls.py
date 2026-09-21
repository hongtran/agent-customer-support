"""Turn persisted image references into short-lived URLs for a response.

Shared by the chat turn (`Coordinator._finish`) and the admin conversation viewer, which
re-renders stored history. Both degrade instead of raising: a signing problem costs a
picture, never the text around it.
"""

import logging

from agent_customer_support import doc_images
from agent_customer_support.models import AttachmentRef, StoredAttachment
from agent_customer_support.stores.attachment_store import AttachmentStore
from agent_customer_support.stores.doc_image_store import DocImageStore

logger = logging.getLogger(__name__)


async def resolve_doc_images(store: DocImageStore, text: str) -> str:
    """Swap `[[img:…]]` markers for presigned URLs.

    Persisted turns keep markers, never URLs — a presigned URL expires, so storing one
    would archive a dead link and feed ~500 characters of signature into the transcript
    the LLM re-reads next turn. Re-rendering history is therefore just re-signing.
    On failure every marker is stripped, so no half-resolved marker reaches the reader.
    """
    try:
        urls = {}
        for _kind, slug, name in doc_images.markers_in(text):
            urls[(slug, name)] = await store.presign(slug, name)
        if not urls:
            return text
        return doc_images.presign_markers(text, lambda s, n: urls[(s, n)])
    except Exception as exc:  # noqa: BLE001 - degrade, never break the text
        logger.warning("doc image presign failed, returning text without images: %s", exc)
        return doc_images.strip(text)


async def presign_attachments(
    store: AttachmentStore, stored: list[StoredAttachment]
) -> list[AttachmentRef]:
    """Signed URLs for a turn's uploaded images. All-or-nothing: `[]` on any failure."""
    if not stored:
        return []
    try:
        return [await store.presign(s) for s in stored]
    except Exception as exc:  # noqa: BLE001 - degrade, never break the text
        logger.warning("presign failed, returning without image urls: %s", exc)
        return []

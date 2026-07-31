import base64
import contextlib
import logging

import aioboto3
from botocore.exceptions import ClientError

from agent_customer_support.config import get_settings
from agent_customer_support.models import Attachment, AttachmentRef, StoredAttachment

logger = logging.getLogger(__name__)

# media_type -> file extension, for a human-readable key. Kept deliberately small;
# Attachment.media_type is already validated against ALLOWED_IMAGE_MEDIA_TYPES.
_EXT = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}


def _session() -> aioboto3.Session:
    return aioboto3.Session()


@contextlib.asynccontextmanager
async def get_client():
    """S3 client, mirroring stores/dynamo.get_resource so both AWS clients are
    configured the same way (endpoint override for LocalStack, shared region)."""
    s = get_settings()
    async with _session().client(
        "s3",
        endpoint_url=s.s3_endpoint_url,
        region_name=s.aws_region,
    ) as s3:
        yield s3


class AttachmentStore:
    """Image bytes live in S3; only the key is persisted on the conversation turn.

    Conversation records are a single DynamoDB item rewritten on every turn, and
    DynamoDB caps items at 400 KB — base64 inflates by 4/3, so a ~300 KB screenshot
    was enough to fail the write and take the reply down with it. Bytes belong here.
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self.bucket = cfg.s3_bucket_attachments

    async def init(self) -> None:
        if not get_settings().s3_auto_create_bucket:
            return
        cfg = get_settings()
        async with get_client() as s3:
            try:
                await s3.create_bucket(
                    Bucket=self.bucket,
                    CreateBucketConfiguration={"LocationConstraint": cfg.aws_region},
                )
            except ClientError as e:
                code = e.response["Error"]["Code"]
                # Already ours, or already exists in this region — both are fine.
                if code not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
                    raise
            # Screenshots routinely contain customer data, so the bucket must never be
            # world-readable; reads go through presigned URLs instead. Not all S3-
            # compatible endpoints implement this call, and it is a hardening step
            # rather than a correctness one, so a failure only warns.
            try:
                await s3.put_public_access_block(
                    Bucket=self.bucket,
                    PublicAccessBlockConfiguration={
                        "BlockPublicAcls": True,
                        "IgnorePublicAcls": True,
                        "BlockPublicPolicy": True,
                        "RestrictPublicBuckets": True,
                    },
                )
            except ClientError as e:
                logger.warning("could not set public access block on %s: %s", self.bucket, e)

    def _key(self, conversation_id: str, turn_id: str, index: int, media_type: str) -> str:
        # Prefixed by conversation so lifecycle rules and per-customer purges are
        # plain prefix operations.
        return f"conversations/{conversation_id}/{turn_id}/{index}.{_EXT.get(media_type, 'bin')}"

    async def put(
        self, conversation_id: str, turn_id: str, index: int, attachment: Attachment
    ) -> StoredAttachment:
        raw = base64.b64decode(attachment.data)
        key = self._key(conversation_id, turn_id, index, attachment.media_type)
        async with get_client() as s3:
            await s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=raw,
                ContentType=attachment.media_type,
            )
        return StoredAttachment(
            kind=attachment.kind,
            media_type=attachment.media_type,
            s3_key=key,
            size_bytes=len(raw),
        )

    async def presign(self, stored: StoredAttachment) -> AttachmentRef:
        """Short-lived GET URL. Signing is local — no network round trip — so this
        is cheap enough to do per attachment on every response."""
        async with get_client() as s3:
            url = await s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": stored.s3_key},
                ExpiresIn=get_settings().s3_presign_expiry_seconds,
            )
        return AttachmentRef(kind=stored.kind, media_type=stored.media_type, url=url)

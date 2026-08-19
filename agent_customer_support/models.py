import base64
from binascii import Error as BinasciiError
from datetime import datetime, UTC
from typing import Annotated, Literal
from uuid import uuid4
from pydantic import BaseModel, Field, StringConstraints, field_validator


def _now() -> datetime:
    return datetime.now(UTC)


# ---- Flow ----


class FlowTransition(BaseModel):
    when: str
    goto: str


class FlowStep(BaseModel):
    id: str
    say: str
    next: list[FlowTransition] = Field(default_factory=list)


class FlowOutcome(BaseModel):
    type: Literal["success", "escalate"]
    say: str | None = None
    reason: str | None = None


class Flow(BaseModel):
    id: str
    title: str
    application: str
    scope: str = "global"
    version: int = 1
    language: str = "vi"
    triggers: list[str] = Field(default_factory=list)
    steps: list[FlowStep] = Field(default_factory=list)
    outcomes: dict[str, FlowOutcome] = Field(default_factory=dict)


# ---- Customer ----

Role = Literal["admin", "user"]

# customer_id doubles as the login username, a JWT `sub`, a DynamoDB partition key and a
# URL path segment, so it is constrained rather than trusted. Shared as a type so the
# admin create-request enforces the identical rule and a bad id is a 422 at the request
# boundary, not a 500 raised from inside a handler building a CustomerProfile.
CustomerId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9._-]{1,64}$")]


class CustomerProfile(BaseModel):
    customer_id: CustomerId
    name: str
    enabled_applications: list[str] = Field(default_factory=list)
    config_notes: str | None = None
    # bcrypt output only — a plaintext password is never persisted. None means this
    # profile has no credentials and cannot log in, which is what every row created
    # before authentication existed looks like.
    password_hash: str | None = None
    role: Role = "user"


# ---- Attachments ----


# Attachments exist in three shapes, deliberately kept as separate types:
#   Attachment       - inbound on the request, carries the actual bytes, fed to the LLM
#   StoredAttachment - what lands in DynamoDB: an S3 key, never the bytes
#   AttachmentRef    - what goes back to the UI: a short-lived presigned URL
# Splitting them is what makes it structurally impossible to persist base64 into a
# conversation record again — a 300 KB screenshot used to blow DynamoDB's 400 KB item
# limit and take the whole (already-generated) reply down with it.

ALLOWED_IMAGE_MEDIA_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})


class Attachment(BaseModel):
    kind: Literal["image"]
    media_type: str  # image/png | image/jpeg | image/webp
    data: str  # base64-encoded bytes

    @field_validator("media_type")
    @classmethod
    def _known_media_type(cls, v: str) -> str:
        if v not in ALLOWED_IMAGE_MEDIA_TYPES:
            raise ValueError(f"unsupported media_type {v!r}")
        return v

    @field_validator("data")
    @classmethod
    def _valid_base64(cls, v: str) -> str:
        # Without this an unparseable blob travels all the way to the LLM before
        # anything notices. Reject at the model boundary instead (422).
        try:
            base64.b64decode(v, validate=True)
        except (BinasciiError, ValueError) as exc:
            raise ValueError(f"data is not valid base64: {exc}") from exc
        return v

    @property
    def decoded_size(self) -> int:
        """Decoded byte count, computed from the base64 length rather than by
        decoding — the point is to size-check an upload without materialising it."""
        s = self.data.rstrip("=")
        return len(s) * 3 // 4


class StoredAttachment(BaseModel):
    kind: Literal["image"]
    media_type: str
    s3_key: str
    size_bytes: int


class AttachmentRef(BaseModel):
    kind: Literal["image"]
    media_type: str
    url: str  # presigned GET, expires per s3_presign_expiry_seconds


# ---- Conversation ----


class Turn(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    role: Literal["user", "assistant"]
    content: str
    attachments: list[StoredAttachment] = Field(default_factory=list)
    ts: datetime = Field(default_factory=_now)


class Conversation(BaseModel):
    conversation_id: str
    customer_id: str
    turns: list[Turn] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)


# ---- Request backlog ----


class RequestRecord(BaseModel):
    id: str
    customer_id: str
    type: Literal["feature", "bug", "how_to_missing"]
    summary: str
    application: str | None = None
    transcript: str = ""
    created_at: datetime = Field(default_factory=_now)


# ---- Q&A learning loop ----


class QARecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    question: str
    answer: str = ""
    status: Literal["pending", "approved", "rejected", "archived"] = "pending"
    source: Literal["cannot_answer", "feedback", "manual"]
    application: str | None = None
    customer_id: str | None = None
    conversation_id: str | None = None
    feedback_message_id: str | None = None
    bad_answer: str | None = None
    transcript: str = ""
    qdrant_point_id: str | None = None
    indexed_at: datetime | None = None
    approved_by: str | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


# ---- Session ----


class SessionState(BaseModel):
    conversation_id: str
    active_flow_id: str | None = None
    current_step_id: str | None = None
    pending: Literal["verify_issue", "knowledge_clarify"] | None = None
    pending_context: dict | None = None
    selected_applications: list[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=_now)


# ---- Channel I/O ----


class ChatRequest(BaseModel):
    # No customer_id: identity comes from the access token, never from the body. A
    # client-supplied tenant id was the whole vulnerability this replaced — it feeds
    # the Qdrant application scoping filter and keys the conversation store.
    conversation_id: str
    message: str
    attachments: list[Attachment] = Field(default_factory=list)
    applications: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    citations: list[str] = Field(default_factory=list)
    escalated: bool = False
    message_id: str = ""
    # The *user* turn's images, echoed back with presigned URLs so the widget can
    # render what was just uploaded. Same shape a history endpoint would return.
    attachments: list[AttachmentRef] = Field(default_factory=list)


# ---- Agent contract ----


class AgentResult(BaseModel):
    action: Literal["reply", "route"] = "reply"
    reply: str = ""
    routed_to: Literal["knowledge", "flow", "escalate", "out_of_scope"] | None = None
    resolved: bool | None = None
    # True when the turn was refused as unrelated to CenLab. Distinct from a plain
    # resolved=False miss on purpose: a miss escalates to Zalo and writes backlog/QA
    # records, which off-topic chatter must never do.
    out_of_scope: bool = False
    suspected_bug: bool = False
    evidence_complete: bool = False
    evidence: dict | None = None
    escalated: bool = False
    new_session: SessionState | None = None
    citations: list[str] = Field(default_factory=list)

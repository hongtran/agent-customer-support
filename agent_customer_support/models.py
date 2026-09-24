import base64
import re
from binascii import Error as BinasciiError
from datetime import datetime, UTC
from typing import Annotated, Literal
from uuid import uuid4
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
)


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


_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def _normalize_email(v: object) -> object:
    """Trim; an empty string means "no manager" (an emptied form field clears it)."""
    if not isinstance(v, str):
        return v
    v = v.strip()
    if not v:
        return None
    if not _EMAIL_RE.fullmatch(v):
        raise ValueError(f"invalid email {v!r}")
    return v


def _blank_to_none(v: object) -> object:
    """Trim; an empty string means "not set" (an emptied form field clears it)."""
    if not isinstance(v, str):
        return v
    return v.strip() or None


# Types rather than field validators so the admin API models share the same checks.
ManagerEmail = Annotated[str | None, BeforeValidator(_normalize_email)]
# A MantisBT username (the login name, not the display name).
MantisUserName = Annotated[str | None, BeforeValidator(_blank_to_none)]


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
    # Questions per day (Vietnam time). None = unlimited, which is what every row created
    # before rate limiting existed looks like. Admins are never limited. The running count
    # lives in UsageStore, not here — see there for why.
    daily_question_limit: int | None = Field(default=None, ge=0)
    # The CenLab employee who manages this customer. The email is CC'd on the handoff
    # mail for a verified bug or an unanswered question; the MantisBT username is the
    # assignee (`handler`) of a verified-bug ticket. A name, not the numeric id, so CS
    # can read in the form who manages the customer. Both optional: None = nobody.
    manager_email: ManagerEmail = None
    mantis_handler_name: MantisUserName = None


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


# ---- Contact ----


class ContactInfo(BaseModel):
    """What the user left for CS to call back, parsed by `contact.parse`.

    Stored on the conversation and the backlog row, never on `CustomerProfile`: one
    customer account is shared by many people at a company, so the person to call
    is a property of this handoff, not of the account. `raw` is the whole message so
    CS also sees anything the regexes could not name ("gọi sau 5h chiều").
    """

    phone: str | None = None
    email: str | None = None
    raw: str = ""

    @property
    def found(self) -> bool:
        return bool(self.phone or self.email)


# ---- Conversation ----


CONVERSATION_TITLE_CHARS = 120


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
    # Left by the user after a handoff (see Coordinator._attach_contact). None until then.
    contact: ContactInfo | None = None
    # Summary for the admin list, derived from `turns` by `refresh_summary` on every
    # write. `customer_id` + `updated_at` key the table's by-customer index, so
    # `updated_at` must never be written as null — DynamoDB rejects a NULL index key.
    title: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    turn_count: int = 0

    def refresh_summary(self) -> None:
        """Recompute the summary fields from `turns`. The one place they are derived,
        so the store and the backfill script cannot disagree."""
        self.turn_count = len(self.turns)
        first_user = next((t.content for t in self.turns if t.role == "user"), None)
        if first_user is not None:
            self.title = first_user.strip()[:CONVERSATION_TITLE_CHARS]
        if self.turns:
            self.created_at = self.turns[0].ts
            self.updated_at = self.turns[-1].ts
        else:
            self.created_at = self.created_at or _now()
            self.updated_at = _now()


class ConversationSummary(BaseModel):
    """One row of the admin conversation list: the index projection, no turns."""

    conversation_id: str
    title: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    turn_count: int = 0


# ---- Request backlog ----


class RequestRecord(BaseModel):
    id: str
    customer_id: str
    type: Literal["feature", "bug", "how_to_missing"]
    summary: str
    # Ticket title as filed in MantisBT. Only a verified bug has one; rows written
    # before tickets existed, and the other request types, leave it None.
    title: str | None = None
    application: str | None = None
    transcript: str = ""
    # Set when the MantisBT issue was created; None means "not configured" or "the
    # create failed and CS was told to file it by hand" — the row exists either way.
    mantis_issue_id: int | None = None
    mantis_issue_url: str | None = None
    # Filled in later by RequestBacklog.set_contact when the user answers the contact ask.
    contact: ContactInfo | None = None
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


# ---- Answer feedback ----


class FeedbackRecord(BaseModel):
    """A customer's like/dislike on one assistant message.

    One item per message, keyed by ``message_id``: a message belongs to one conversation,
    which belongs to one customer, so there is only ever one voter. A new vote overwrites
    the old one, and clearing a vote deletes the item, so counts stay correct.
    """

    message_id: str
    conversation_id: str
    customer_id: str
    signal: Literal["up", "down"]
    # Copies of the text at vote time, so a feedback list is readable without loading
    # each conversation. `answer` is the persisted form, with [[img:…]] markers.
    question: str = ""
    answer: str = ""
    # Time of the latest vote — a changed vote replaces the item.
    created_at: datetime = Field(default_factory=_now)


# ---- Session ----


_SLOT_LABELS: dict[str, str] = {
    "module": "màn hình/menu",
    "version": "phiên bản/môi trường",
    "steps": "các bước tái hiện",
    "expected": "kết quả mong đợi",
    "actual": "kết quả thực tế",
    "occurred_at": "thời điểm xảy ra",
}

# The slot names as a type, for `VerificationDecision.ask_for`. Must match the keys of
# `_SLOT_LABELS`; a test keeps the two in sync.
SlotName = Literal["module", "version", "steps", "expected", "actual", "occurred_at"]

# What a ticket has to say to be worth an engineer's time. The other three slots are
# useful, never blocking -- a version number is not worth losing the report over.
_REQUIRED_SLOTS = ("module", "actual", "steps")


class BugSlots(BaseModel):
    """The fixed list of facts a bug ticket needs, and how much of it we have.

    Both a domain type (it is persisted inside `SessionState.pending_context`
    between turns) and the wire format of `VerificationDecision.slots` -- hence the
    Vietnamese descriptions and `extra="forbid"`, which belong to the LLM side. It
    lives here rather than in `llm/schemas.py` because the session is what carries
    it across turns, and `models` importing the LLM layer would invert the
    dependency the rest of the package keeps.

    Every field is required with "" as the no-value case, the same rule as
    `CitedSource.section`: a defaulted field is dropped from `required` under
    OpenAI strict mode. `empty()` is how Python builds one, since that leaves no
    default for the schema to lose.

    There is no `has_image` here on purpose. Prior turns' screenshots are not
    re-sent to the verification model, so whether one has ever arrived is a fact
    only Python holds -- it lives on `VerifyContext` instead, where nothing invites
    the model to overwrite it.

    `module` is NOT `VerifyContext.application`, and the two names are kept apart on
    purpose. An *application* is what the customer bought and what scopes retrieval
    ("Lấy mẫu - Quan trắc", a slug in Qdrant); a *module* is one level down inside it
    -- the menu, screen or page the user was on. Only the application reaches the
    MantisBT `category` and the Qdrant filter, so conflating them would scope a search
    to a screen name that matches nothing.
    """

    model_config = ConfigDict(extra="forbid")

    # Deliberately loose. An earlier wording demanded a name "inside" the application
    # and not the application itself; the model then read "Quy chuẩn/Tiêu chuẩn" as a
    # parent area, left the slot empty, and asked for the screen name every turn.
    module: str = Field(
        description=(
            "Tên menu/màn hình/trang/module nơi xảy ra lỗi, CHÉP NGUYÊN VĂN như người dùng "
            "nói hoặc như thấy trên ảnh chụp (ví dụ: 'Quy chuẩn/Tiêu chuẩn', 'Danh sách "
            "phiếu yêu cầu'). MỘT tên là ĐỦ — không đòi tên chi tiết hơn. Rỗng nếu chưa "
            "có tên nào."
        )
    )
    version: str = Field(description="Phiên bản hoặc môi trường (web/desktop). Rỗng nếu chưa biết.")
    steps: str = Field(
        description="Các bước người dùng đã làm để gặp lỗi, mỗi bước một dòng. Rỗng nếu chưa biết."
    )
    expected: str = Field(description="Kết quả người dùng mong đợi. Rỗng nếu chưa biết.")
    actual: str = Field(
        description="Điều thực sự xảy ra, kèm thông báo lỗi nếu có. Rỗng nếu chưa biết."
    )
    occurred_at: str = Field(description="Thời điểm hoặc tần suất xảy ra. Rỗng nếu chưa biết.")

    @classmethod
    def empty(cls) -> "BugSlots":
        return cls(module="", version="", steps="", expected="", actual="", occurred_at="")

    def merge(self, update: "BugSlots") -> "BugSlots":
        """Apply a turn's reading of the slots without ever blanking a filled one.

        The model sees the whole conversation and re-states every slot each turn, so
        a slot it leaves empty means "nothing new here", never "forget that". A
        non-empty value wins, which lets the user correct themselves.
        """
        return BugSlots(
            **{
                name: (getattr(update, name).strip() or getattr(self, name).strip())
                for name in _SLOT_LABELS
            }
        )

    def missing_slots(self) -> list[str]:
        """Names of the required slots still empty, for code that reasons about them."""
        return [n for n in _REQUIRED_SLOTS if not getattr(self, n).strip()]

    def missing(self) -> list[str]:
        """Labels of the required slots still empty, for the ticket's own note."""
        return [_SLOT_LABELS[n] for n in self.missing_slots()]

    def describe(self) -> str:
        """The filled slots as a readable block for the ticket body."""
        return "\n".join(
            f"- {_SLOT_LABELS[name]}: {value.strip()}"
            for name in _SLOT_LABELS
            if (value := getattr(self, name)).strip()
        )


class VerifyContext(BaseModel):
    """`SessionState.pending_context` while a bug is being verified.

    A typed view over the dict rather than a replacement for it: the dict is what
    Redis already holds, and sessions written before this type existed carry only
    `application`, `summary` and `since_turn`. Every field therefore has a default,
    and extra keys are ignored, so a live session mid-flow keeps working across a
    deploy instead of failing validation and losing the collected evidence.
    """

    # The APPLICATION slug -- what the customer bought, what scopes Qdrant and what
    # the ticket is filed under. The screen inside it is `slots.module`; see BugSlots
    # on why the two are not the same field.
    application: str | None = None
    summary: str = ""
    # Index of the turn the bug was first suspected on. The ticket attaches
    # screenshots from there onward only, so an unrelated image sent earlier in the
    # same conversation never lands on it.
    since_turn: int = 0
    slots: BugSlots = Field(default_factory=BugSlots.empty)
    has_image: bool = False
    # The ask-once guard (IssueVerificationAgent._guard). `asked_last` is what the
    # previous reply asked for, so this turn's message can be read as the answer to it;
    # `ask_counts` is every slot ever asked, so none is asked twice.
    asked_last: list[str] = Field(default_factory=list)
    ask_counts: dict[str, int] = Field(default_factory=dict)
    report: dict | None = None
    # The doc check (IssueVerificationAgent._doc_check) runs once per bug, before any
    # slot is asked. `query` is KnowledgeAgent's contextualized search query when the
    # bug came from there (empty on a direct triage route); `doc_expected` is what the
    # guides say should happen, kept as short text so later turns and the ticket can
    # use it without storing whole passages in Redis.
    query: str = ""
    doc_checked: bool = False
    doc_expected: str = ""


class SessionState(BaseModel):
    conversation_id: str
    pending: Literal["verify_issue", "knowledge_clarify", "collect_contact"] | None = None
    pending_context: dict | None = None
    selected_applications: list[str] = Field(default_factory=list)
    # Routing counters. They live on the session because every one of them bounds a
    # loop that spans turns, and they are what `routing.next_step` reads to turn a
    # cap into a handoff. New fields with defaults on purpose: a session written
    # before they existed must still load.
    clarify_count: int = 0
    verify_turns: int = 0
    # Verification has already ruled this conversation a user error. Knowledge
    # suspecting a bug again is the two of them disagreeing, not new evidence.
    user_error_seen: bool = False
    updated_at: datetime = Field(default_factory=_now)


# ---- Channel I/O ----


# ---- Citations ----


class Citation(BaseModel):
    """One source the composed answer declared it used, after validation.

    Not "everything retrieved" — that is what `RagClient.search` returns and what this
    field used to carry. A Citation exists only because the composer named it AND the id
    it named was in this turn's catalog (see `citations.select`).

    **No filename appears here.** The source document's name is internal, and this type
    exists to be shown to the customer, so there is deliberately no field for it to sit
    in — not a hidden one, not an unused one. `label` is built from the section heading,
    the application display name, or a fixed constant, and `citations.py` holds no
    filename it could put there.

    `doc_id` is the raw identifier — a `source_doc_id`/`doc_id` UUID for a guide,
    `qa:<id>` for a CS-verified Q&A record, or the pseudo-id `quy_trinh_chung` for the
    always-on process block. It is opaque, and it is a handle for traces, never display.
    """

    doc_id: str
    # What the widget prints.
    label: str
    # The heading the answer drew from, when one was matched. Kept separate from `label`
    # so a caller can tell "this row names a section" from "this row is only an
    # application" — `label` alone cannot distinguish the two.
    section: str | None = None
    application: str | None = None
    kind: Literal["guide", "qa", "process"]
    confidence: float = 0.0


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
    citations: list[Citation] = Field(default_factory=list)
    escalated: bool = False
    message_id: str = ""
    # The *user* turn's images, echoed back with presigned URLs so the widget can
    # render what was just uploaded. Same shape a history endpoint would return.
    attachments: list[AttachmentRef] = Field(default_factory=list)
    # Questions left today after this one; None = no daily limit (or an admin).
    questions_remaining: int | None = None


# ---- Agent contract ----


class AgentResult(BaseModel):
    """What one agent hands back to the driver.

    The three `*_status` fields below are the whole routing contract: each agent
    reports what it decided as a validated value, and `routing.next_step` reads
    them without re-parsing any prose. They replaced a pair of booleans and a
    tri-state `resolved` that encoded four outcomes between them.
    """

    reply: str = ""
    routed_to: Literal["knowledge", "issue_verification", "escalate", "out_of_scope"] | None = None
    # What KnowledgeAgent decided. "answer" is a real answer; "clarify" is a question
    # back to the user; "no_answer" is a miss already logged to the backlog; and
    # "suspected_bug" says the guides claim the feature works, so the report deserves
    # verification.
    knowledge_status: Literal["answer", "clarify", "no_answer", "suspected_bug"] | None = None
    # What IssueVerificationAgent decided. "user_error" means the feature behaved
    # correctly and the user needs an explanation, not a ticket.
    verify_outcome: Literal["need_more_info", "user_error", "bug_confirmed"] | None = None
    # Why the driver moved the turn here, from `routing.next_step`. Logged on the
    # step's trace span so a route can be explained without replaying the turn.
    handoff_reason: str | None = None
    # True when the turn was refused as unrelated to CenLab. Distinct from a plain
    # knowledge miss on purpose: a miss escalates to Zalo and writes backlog/QA
    # records, which off-topic chatter must never do.
    out_of_scope: bool = False
    evidence: dict | None = None
    escalated: bool = False
    # Why the handoff happened ("user requested human", "verified bug", a flow outcome's
    # reason...). Set wherever `escalated` is set; read by the contact follow-up so the
    # second CS notification can say which handoff the contact belongs to.
    escalation_reason: str | None = None
    # Records the handoff created, so a contact given on a later turn can be attached to
    # them: {"backlog_id", "mantis_issue_id", "mantis_issue_url"}. Only the verified-bug
    # path fills this.
    escalation_refs: dict | None = None
    citations: list[Citation] = Field(default_factory=list)
    # The text of the passages `citations` point at, carried only from KnowledgeAgent to
    # the output guardrail so the grounding judge can see what the answer claimed to be
    # based on without re-running retrieval. Excluded from serialisation on purpose:
    # Coordinator._traced dumps every AgentResult into a Langfuse span, and full passage
    # text would bloat every trace for a value nothing downstream reads.
    source_passages: list[str] = Field(default_factory=list, exclude=True)

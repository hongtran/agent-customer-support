from datetime import datetime, UTC
from typing import Literal
from pydantic import BaseModel, Field


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
    module: str
    scope: str = "global"
    version: int = 1
    language: str = "vi"
    triggers: list[str] = Field(default_factory=list)
    steps: list[FlowStep] = Field(default_factory=list)
    outcomes: dict[str, FlowOutcome] = Field(default_factory=dict)


# ---- Customer ----


class CustomerProfile(BaseModel):
    customer_id: str
    name: str
    enabled_modules: list[str] = Field(default_factory=list)
    config_notes: str | None = None


# ---- Conversation ----


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
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
    module: str | None = None
    transcript: str = ""
    created_at: datetime = Field(default_factory=_now)


# ---- Session ----


class SessionState(BaseModel):
    conversation_id: str
    active_flow_id: str | None = None
    current_step_id: str | None = None
    updated_at: datetime = Field(default_factory=_now)


# ---- Channel I/O ----


class ChatRequest(BaseModel):
    customer_id: str
    conversation_id: str
    message: str


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    citations: list[str] = Field(default_factory=list)
    escalated: bool = False

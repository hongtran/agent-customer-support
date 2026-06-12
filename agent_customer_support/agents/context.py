from dataclasses import dataclass, field
from typing import Any

from agent_customer_support.models import (
    Attachment, Conversation, CustomerProfile, SessionState,
)


@dataclass
class TurnContext:
    customer: CustomerProfile
    session: SessionState
    conversation: Conversation
    message: str
    attachments: list[Attachment] = field(default_factory=list)
    transcript: str = ""
    # shared service handles (typed Any to avoid import cycles with stores)
    rag: Any = None
    flow_store: Any = None
    backlog: Any = None
    escalator: Any = None

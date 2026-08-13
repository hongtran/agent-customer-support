from dataclasses import dataclass, field
from typing import Any

from agent_customer_support.models import (
    Attachment,
    Conversation,
    CustomerProfile,
    SessionState,
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
    doc_images: Any = None
    flow_store: Any = None
    backlog: Any = None
    qa_store: Any = None
    escalator: Any = None

    def as_messages(self) -> list[dict]:
        """Multi-turn message list from conversation history + current plain-text message.

        Use this for agents that route or collect information across turns (Triage,
        Verification). For agents that also need vector search (Knowledge), call
        _contextualize() first to resolve pronouns before using the standalone query.
        """
        msgs = [{"role": t.role, "content": t.content} for t in self.conversation.turns]
        msgs.append({"role": "user", "content": self.message})
        return msgs

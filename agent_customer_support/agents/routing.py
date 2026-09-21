"""Which agent runs next, as a pure function.

Every decision about *where a turn goes* lives here, and nothing in this module
does I/O, calls an LLM, or imports an agent or a store. That is the whole point:
`next_step` is a table of rules over a small frozen snapshot, so each branch is
one assertion in a test with no mocks at all. `Coordinator._route` is the other
half -- the driver that executes a step, updates the session, and folds the
result back into the snapshot the next decision reads.

The limits are the second reason this module exists. Before it the pipeline had
none: an evidence collection that never completed kept `pending="verify_issue"`
for as long as the Redis session lived, and nothing bounded a clarify loop or a
knowledge/verification ping-pong. A user must never reach a dead end, so every
counter here has a rule that turns the cap into a handoff rather than a wait.

`MAX_HOPS` is the backstop for the rules themselves. The longest legitimate
path is triage -> knowledge -> issue_verification -> knowledge -> reply (four
hops), so a run that reaches six has found a cycle these rules did not
anticipate, and escalating is the only honest answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Literal

from agent_customer_support.models import AgentResult, SessionState

Step = Literal[
    "triage",
    "knowledge",
    "issue_verification",
    "escalate",
    "out_of_scope",
    "attach_contact",
    "file_ticket",
    "reply",
]

#: Steps that end the turn. Everything else runs an agent and loops.
TERMINAL: frozenset[str] = frozenset(
    {"escalate", "out_of_scope", "attach_contact", "file_ticket", "reply"}
)

MAX_HOPS = 6
MAX_CLARIFY = 2
MAX_VERIFY_TURNS = 4

# Letting go of a pending flow must not cost an LLM call, so this is a regex and
# not a second triage pass. It is deliberately conservative: a false positive
# throws away collected evidence, while a miss only means the user repeats
# themselves. Hence the bare verbs are anchored to the start of the message --
# "thôi" in the middle of a sentence ("mình chỉ cần in thôi") is not a cancel.
_CANCEL_RE = re.compile(
    r"^\s*(thôi|khoan|dừng|bỏ qua|hủy|huỷ)\b"
    r"|\bbỏ qua (đi|nhé|vụ|chuyện)"
    r"|\bquên (đi|nó|vụ|chuyện)"
    r"|\bkhông cần (nữa|hỗ trợ nữa|báo nữa)"
    r"|\b(câu hỏi|vấn đề|việc|chuyện) khác\b"
    r"|\bchuyển (chủ đề|sang câu)",
    re.IGNORECASE,
)


def wants_cancel(message: str) -> bool:
    """True when the user is dropping the flow they are in, not answering it."""
    return bool(_CANCEL_RE.search(message or ""))


@dataclass(frozen=True)
class RouteState:
    """Everything `next_step` is allowed to look at.

    Deliberately flat strings rather than the models themselves: a rule that could
    reach into a store or an `AgentResult`'s reply text would stop being checkable
    at a glance. `start` and `after` are the only two ways one is built, so the
    driver cannot accidentally hand the rules a half-updated snapshot.
    """

    pending: str | None = None
    contact_found: bool = False
    triage_target: str | None = None
    knowledge_status: str | None = None
    verify_outcome: str | None = None
    clarify_count: int = 0
    verify_turns: int = 0
    user_error_seen: bool = False
    hop_count: int = 0

    @classmethod
    def start(cls, session: SessionState, *, contact_found: bool = False) -> RouteState:
        """The snapshot at the top of a turn, before any agent has run."""
        return cls(
            pending=session.pending,
            contact_found=contact_found,
            clarify_count=session.clarify_count,
            verify_turns=session.verify_turns,
            user_error_seen=session.user_error_seen,
        )

    def after(self, result: AgentResult, session: SessionState) -> RouteState:
        """Fold one executed step back in and count the hop.

        `pending` is cleared because a step that ran has consumed whatever resumed
        it: the next decision must come from what that step returned, or the rules
        would send the turn straight back into the same agent. The counters are
        re-read from the session rather than tracked here, so there is exactly one
        place they live.
        """
        return replace(
            self,
            pending=None,
            contact_found=False,
            # Triage sets this once and later steps leave it None; keeping the last
            # value means a knowledge result that resolves nothing still falls back
            # to the route triage chose.
            triage_target=result.routed_to or self.triage_target,
            knowledge_status=result.knowledge_status,
            verify_outcome=result.verify_outcome,
            clarify_count=session.clarify_count,
            verify_turns=session.verify_turns,
            user_error_seen=session.user_error_seen,
            hop_count=self.hop_count + 1,
        )


def next_step(s: RouteState) -> tuple[Step, str]:
    """The next step and the reason for it. Pure: no I/O, no LLM, no mutation.

    The reason is not decoration. It is logged on the step's trace span, and for
    every rule that lands on `escalate` it is also the handoff reason CS reads, so
    a conversation can be explained after the fact without replaying it.

    Order matters. Reading top to bottom: safety first, then a flow the user is
    already in, then what the last agent decided, then what triage chose.
    """
    if s.hop_count >= MAX_HOPS:
        return "escalate", "hop limit"

    # The turn after a handoff, and the user left a phone number or an email.
    if s.contact_found:
        return "attach_contact", "contact given"

    # A flow already in progress owns the turn (the driver has already checked
    # whether the user is cancelling it).
    if s.pending == "verify_issue":
        # Checked before the agent runs, so the cap holds even if the model would
        # have asked another question. Whatever was collected becomes the ticket.
        if s.verify_turns >= MAX_VERIFY_TURNS:
            return "file_ticket", "verify turn cap"
        return "issue_verification", "resume verification"
    if s.pending == "knowledge_clarify":
        return "knowledge", "clarify answered"

    if s.verify_outcome == "bug_confirmed":
        return "file_ticket", "verified bug"
    if s.verify_outcome == "user_error":
        # Not a bug: the user needs to be told how the feature actually works, which
        # is knowledge's job, not a ticket.
        return "knowledge", "not a bug, explain usage"
    if s.verify_outcome == "need_more_info":
        return "reply", "collecting evidence"

    if s.knowledge_status == "suspected_bug":
        # Verification already ruled this conversation a user error once. Knowledge
        # disagreeing a second time is the two agents ping-ponging, not new
        # information, so stop paying for it and hand over.
        if s.user_error_seen:
            return "escalate", "bug loop"
        return "issue_verification", "suspected bug"
    if s.knowledge_status == "clarify":
        if s.clarify_count >= MAX_CLARIFY:
            return "escalate", "clarify limit"
        return "reply", "need clarification"
    if s.knowledge_status == "no_answer":
        return "escalate", "knowledge unresolved"
    if s.knowledge_status == "answer":
        return "reply", "answered"

    if s.triage_target == "escalate":
        return "escalate", "user requested human"
    if s.triage_target == "issue_verification":
        return "issue_verification", "reported malfunction"
    if s.triage_target == "out_of_scope":
        return "out_of_scope", "off topic"
    if s.triage_target == "knowledge":
        return "knowledge", "question"

    return "triage", "new turn"

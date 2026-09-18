import re

from agent_customer_support import doc_images
from agent_customer_support.agents.prompts import GROUNDING_JUDGE_PROMPT, PROCESS_BLOCK
from agent_customer_support.config import get_settings
from agent_customer_support.llm import complete_structured
from agent_customer_support.llm.schemas import GroundingVerdict

MAX_INPUT_CHARS = 5000

# Limits for the Python delete in `strip_claims`. A span past any of these is a sentence
# or more, and deleting a sentence can silently drop a step the user needed -- that is
# the LLM repair's job, which can reword instead of cut.
_MAX_SPAN_CHARS = 80
_MAX_SPAN_WORDS = 12
# Below this many characters of prose the deletion took the answer with it.
_MIN_REPLY_CHARS = 20
_SENTENCE_END = (".", "!", "?")


def _sources_block(passages: list[str]) -> str:
    return "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages))


def only_minor(claims: list[dict]) -> bool:
    """True when there is at least one claim and none is critical.

    An empty list is False on purpose: `grounded=false` with nothing named is a judge
    that could not point at the problem, and a reply we cannot repair is one we hand off.
    """
    return bool(claims) and all(c.get("severity") == "minor" for c in claims)


def _tidy(text: str) -> str:
    """Whitespace left behind by a deletion: doubled spaces, a space before punctuation,
    trailing spaces at line ends. Nothing else -- this is not a rewrite."""
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" +([,.;:!?])", r"\1", text)
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def strip_claims(reply: str, claims: list[dict]) -> str | None:
    """Delete every minor span from `reply`, or return None if that is not safe.

    All-or-nothing: a half-repaired reply would ship text the judge flagged with no
    record that it was flagged. Each span must be minor, findable EXACTLY once in the
    text as it stands after the earlier deletions (so overlapping spans fail as "not
    found" rather than being skipped), short enough to be a phrase rather than a
    sentence, and outside every image marker. The result must still hold some prose.
    """
    if not only_minor(claims):
        return None
    markers_before = doc_images.markers_in(reply)
    text = reply
    for claim in claims:
        span = claim.get("span") or ""
        core = span.strip()
        if not core:
            return None
        if (
            len(core) > _MAX_SPAN_CHARS
            or len(core.split()) > _MAX_SPAN_WORDS
            or "\n" in core
            or core.endswith(_SENTENCE_END)
        ):
            return None
        if text.count(span) != 1:
            return None
        text = text.replace(span, "", 1)
    if doc_images.markers_in(text) != markers_before:
        return None
    text = _tidy(text)
    if len(doc_images.strip(text)) < _MIN_REPLY_CHARS:
        return None
    return text


class GuardrailAgent:
    name = "guardrail"

    async def check_input(self, message: str) -> dict:
        text = (message or "").strip()
        if not text:
            return {"pass": False, "reason": "empty_input"}
        if len(text) > MAX_INPUT_CHARS:
            return {"pass": False, "reason": "oversized_input"}
        return {"pass": True, "reason": ""}

    async def check_output(self, reply: str, cited_passages: list[str] | None = None) -> dict:
        """Judge whether every claim in `reply` is supported by the sources it cited.

        Grounding only. Tone, scope and prompt-leakage are not judged here — the scope
        gate is triage (see Coordinator._route) and mixing the three into one verdict is
        what made the previous single guardrail prompt hard to tune.

        **No cited passages means no call.** A process-only answer, a clarifying question,
        the canonical no-answer reply and every non-knowledge route arrive here with an
        empty list; there is nothing to check them against, so judging them would flag
        correct replies and spend a call on every single turn. The process block still
        rides in the system prefix for the answers that DO cite a passage, so a reply
        mixing process and passage material is judged against both.

        A failed verdict carries `unsupported_claims` as a list of
        `{span, severity, reason}` dicts: the coordinator reads the severities to choose
        between a Python delete, an LLM repair and a handoff. `reason` joins the per-claim
        reasons for the log line and the eval CSV.
        """
        if not cited_passages:
            return {"pass": True, "reason": ""}

        verdict = complete_structured(
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"NGUỒN ĐÃ DẪN:\n{_sources_block(cited_passages)}"
                        f"\n\nCÂU TRẢ LỜI CẦN KIỂM TRA:\n{reply}"
                    ),
                }
            ],
            system=[PROCESS_BLOCK, {"type": "text", "text": GROUNDING_JUDGE_PROMPT}],
            model=get_settings().model_for("guardrail"),
            schema=GroundingVerdict,
        )
        # Fail OPEN, deliberately the opposite of triage's fail-safe: a judge that
        # could not answer must not be allowed to silence an already-paid-for reply.
        if verdict is None:
            return {"pass": True, "reason": ""}
        if not verdict.grounded:
            claims = [c.model_dump() for c in verdict.unsupported_claims]
            return {
                "pass": False,
                "reason": " | ".join(c["reason"] for c in claims if c["reason"]) or "ungrounded",
                "unsupported_claims": claims,
            }
        return {"pass": True, "reason": ""}

import re

from agent_customer_support import doc_images
from agent_customer_support.agents.prompts import GROUNDING_JUDGE_PROMPT, PROCESS_BLOCK
from agent_customer_support.config import get_settings
from agent_customer_support.llm import complete_structured
from agent_customer_support.llm.schemas import GroundingVerdict

MAX_INPUT_CHARS = 5000

# Limits for the Python edit in `apply_claims`. At most this many words may be REMOVED
# by one claim: past that the judge is rewriting, not trimming, and a rewrite is the
# LLM repair's job. A pure delete (empty replacement) is held to two more guards --
# not a whole sentence, not longer than a phrase -- because deleting a sentence can
# silently drop a step the user needed, where a replacement keeps the sentence alive.
_MAX_REMOVED_WORDS = 12
_MAX_DELETE_CHARS = 80
# Below this many characters of prose the edit took the answer with it.
_MIN_REPLY_CHARS = 20
_SENTENCE_END = (".", "!", "?")
_TOKEN_PUNCT = ".,;:!?()[]\"'“”‘’…"


def _sources_block(passages: list[str]) -> str:
    return "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages))


def _question_block(question: str) -> str:
    """The customer's question, ahead of the sources, in the order the prompt lists them.
    Empty when there is none (an eval row, a test), so no dangling header reaches the judge."""
    question = (question or "").strip()
    return f"CÂU HỎI CỦA KHÁCH HÀNG:\n{question}\n\n" if question else ""


def only_minor(claims: list[dict]) -> bool:
    """True when there is at least one claim and none is major.

    An empty list is False on purpose: `grounded=false` with nothing named is a judge
    that could not point at the problem, and a reply we cannot repair is one we hand off.
    """
    return bool(claims) and all(c.get("severity") == "minor" for c in claims)


def _words(text: str) -> list[str]:
    """Bare words for the subsequence check: punctuation and case are grammar, which a
    replacement is allowed to fix; the words themselves are content, which it is not."""
    out = []
    for tok in text.split():
        core = tok.strip(_TOKEN_PUNCT).casefold()
        if core:
            out.append(core)
    return out


def _is_subsequence(short: list[str], long: list[str]) -> bool:
    it = iter(long)
    return all(any(w == x for x in it) for w in short)


def _safe_edit(span: str, replacement: str) -> bool:
    """Whether replacing `span` with `replacement` removes content without adding any.

    The replacement's words must be a subsequence of the span's words -- same words, same
    order, some left out -- and at least one word must actually go. That is the whole
    guard against the judge becoming a second composer: it can trim and re-punctuate,
    never introduce a step, a button or a condition the answer did not already contain.
    """
    span_words, repl_words = _words(span), _words(replacement)
    if len(repl_words) >= len(span_words):
        return False
    if not _is_subsequence(repl_words, span_words):
        return False
    return len(span_words) - len(repl_words) <= _MAX_REMOVED_WORDS


def _tidy(text: str) -> str:
    """Whitespace left behind by an edit: doubled spaces, a space before punctuation,
    trailing spaces at line ends. Nothing else -- this is not a rewrite."""
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" +([,.;:!?])", r"\1", text)
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def apply_claims(reply: str, claims: list[dict]) -> str | None:
    """Apply every minor claim's replacement to `reply`, or return None if not safe.

    All-or-nothing: a half-repaired reply would ship text the judge flagged with no
    record that it was flagged. Each span must be minor and findable EXACTLY once in the
    text as it stands after the earlier edits (so overlapping spans fail as "not found"
    rather than being skipped). A replacement must pass `_safe_edit`; an empty one is a
    delete, held to the phrase-not-sentence guards above. Image markers must survive
    untouched, and the result must still hold some prose.
    """
    if not only_minor(claims):
        return None
    markers_before = doc_images.markers_in(reply)
    text = reply
    for claim in claims:
        span = claim.get("span") or ""
        replacement = (claim.get("replacement") or "").strip()
        core = span.strip()
        if not core or "\n" in core:
            return None
        if replacement:
            if not _safe_edit(core, replacement):
                return None
        elif (
            len(core) > _MAX_DELETE_CHARS
            or len(_words(core)) > _MAX_REMOVED_WORDS
            or core.endswith(_SENTENCE_END)
        ):
            return None
        if text.count(span) != 1:
            return None
        text = text.replace(span, replacement, 1)
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

    async def check_output(
        self, reply: str, source_passages: list[str] | None = None, question: str = ""
    ) -> dict:
        """Judge whether every claim in `reply` is supported by this turn's sources.

        `question` is the customer's message for this turn. The guides are written for
        every lab, so they never hold the customer's own facts (how many rooms, which
        volumes, how they work today); without the question the judge flags a reply that
        applies a generic step to those facts as invented. It is context, not a source
        of product claims -- the prompt still makes a feature the customer only asked
        about a major claim.

        `source_passages` is every passage the knowledge turn retrieved (guides and Q&A),
        not only the ones the answer cited: a composer that cites the wrong chunk, or
        forgets one it used, must not get its correct claims flagged.

        Grounding only. Tone, scope and prompt-leakage are not judged here — the scope
        gate is triage (see Coordinator._route) and mixing the three into one verdict is
        what made the previous single guardrail prompt hard to tune.

        **No passages means no call.** KnowledgeAgent fills the list only when the answer
        cited at least one real passage. A process-only answer, a clarifying question, the
        canonical no-answer reply and every non-knowledge route arrive here with an empty
        list; there is nothing to check them against, so judging them would flag
        correct replies and spend a call on every single turn. The process block still
        rides in the system prefix for the answers that DO get judged, so a reply
        mixing process and passage material is judged against both.

        A failed verdict carries `unsupported_claims` as a list of
        `{span, replacement, severity, reason}` dicts: the coordinator reads the severities to choose
        between a Python delete, an LLM repair and a handoff. `reason` joins the per-claim
        reasons for the log line and the eval CSV.
        """
        if not source_passages:
            return {"pass": True, "reason": ""}

        verdict = complete_structured(
            messages=[
                {
                    "role": "user",
                    "content": (
                        _question_block(question) + f"NGUỒN:\n{_sources_block(source_passages)}"
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

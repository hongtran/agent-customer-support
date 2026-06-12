import json
import re

from agent_customer_support.agents.context import TurnContext
from agent_customer_support.agents.prompts import (
    KNOWLEDGE_GRADER_PROMPT, KNOWLEDGE_REFORMULATE_PROMPT, KNOWLEDGE_COMPOSE_PROMPT,
)
from agent_customer_support.config import get_settings
from agent_customer_support.llm import complete_text
from agent_customer_support.models import AgentResult

_NO_ANSWER_RE = re.compile(r"\[\[no_answer\]\]")
_BUG_RE = re.compile(r"\[\[suspected_bug:([a-zA-Z0-9_\-]+)\]\]")

HIGH = 0.80
LOW = 0.50
MIN_SUBSTANTIAL_CHARS = 200   # high score but shorter than this => grade


def needs_grading(top_confidence: float, passages: list[str]) -> bool:
    """Return True if the LLM grader should be consulted to confirm answer presence."""
    if not passages:
        return False
    if LOW <= top_confidence < HIGH:
        return True
    if top_confidence >= HIGH:
        total = sum(len(p) for p in passages)
        return total < MIN_SUBSTANTIAL_CHARS
    return False  # low score: skip grader, reformulate instead


def parse_markers(text: str) -> tuple[str, str | None, str | None]:
    """Return (clean_text, kind, module) where kind in {None, 'no_answer', 'suspected_bug'}."""
    bug = _BUG_RE.search(text or "")
    if bug:
        clean = _BUG_RE.sub("", text).strip()
        return clean, "suspected_bug", bug.group(1)
    if _NO_ANSWER_RE.search(text or ""):
        return _NO_ANSWER_RE.sub("", text).strip(), "no_answer", None
    return (text or "").strip(), None, None


def _passages_block(passages: list[str]) -> str:
    return "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages))


class KnowledgeAgent:
    name = "knowledge"

    async def _grade(self, question: str, passages: list[str]) -> bool:
        """Ask the LLM grader whether the passages actually answer the question."""
        raw = complete_text(
            messages=[{"role": "user",
                       "content": f"CÂU HỎI: {question}\n\nĐOẠN TRÍCH:\n{_passages_block(passages)}"}],
            system=KNOWLEDGE_GRADER_PROMPT,
        )
        try:
            return bool(json.loads(raw).get("answer_present"))
        except (json.JSONDecodeError, TypeError):
            return False  # fail-closed: don't answer if grader is unparseable

    async def _reformulate(self, ctx: TurnContext, passages: list[str]) -> str:
        """Rewrite the user's question using CenLab terminology for a better search."""
        hint = ", ".join(ctx.customer.enabled_modules)
        notes = ctx.customer.config_notes or ""
        raw = complete_text(
            messages=[{"role": "user",
                       "content": f"CÂU HỎI GỐC: {ctx.message}\nMODULE: {hint}\nGHI CHÚ: {notes}"}],
            system=KNOWLEDGE_REFORMULATE_PROMPT,
        )
        return (raw or ctx.message).strip()

    async def _compose(self, question: str, passages: list[str]) -> str:
        """Compose a grounded answer from the retrieved passages."""
        return complete_text(
            messages=[{"role": "user",
                       "content": f"CÂU HỎI: {question}\n\nĐOẠN TRÍCH:\n{_passages_block(passages)}"}],
            system=KNOWLEDGE_COMPOSE_PROMPT,
        )

    async def _present(self, ctx: TurnContext, passages: list[str], conf: float) -> bool:
        """Determine whether the passages are present/relevant enough to attempt composing."""
        if not passages:
            return False
        if needs_grading(conf, passages):
            return await self._grade(ctx.message, passages)
        return conf >= HIGH  # trust high, distrust low

    async def run(self, ctx: TurnContext) -> AgentResult:
        """
        Two-attempt pipeline:
          Attempt 1: search(original) → check presence → compose → inspect marker
          If compose emits [[no_answer]] on attempt 1, fall through to attempt 2.
          Attempt 2: reformulate → search(new_query) → check presence → compose → inspect marker
          If still no_answer (or not present after attempt 2), log + return unresolved.
        suspected_bug on either attempt returns immediately.
        """
        collection = get_settings().product_collection
        query = ctx.message

        for attempt in range(2):
            # Search
            res = await ctx.rag.search(query, collection=collection)
            passages = res.get("passages", []) or []
            conf = res.get("top_confidence", 0.0)
            citations = res.get("citations", []) or []

            present = await self._present(ctx, passages, conf)

            if present:
                composed = await self._compose(ctx.message, passages)
                clean, kind, module = parse_markers(composed)

                if kind == "suspected_bug":
                    return AgentResult(
                        reply=clean, resolved=False, suspected_bug=True,
                        evidence={"module": module, "summary": ctx.message},
                        citations=citations,
                    )

                if kind != "no_answer":
                    # Genuine answer found
                    return AgentResult(reply=clean, resolved=True, citations=citations)

                # kind == "no_answer": fall through to attempt 2 (or log if last attempt)

            # If not present OR compose said no_answer:
            if attempt == 0:
                # Reformulate and retry
                query = await self._reformulate(ctx, passages)
            # else: second attempt failed — fall out of loop to log

        # Both attempts exhausted without a real answer
        await ctx.backlog.add(
            customer_id=ctx.customer.customer_id, type="how_to_missing",
            summary=ctx.message, module=None, transcript=ctx.transcript,
        )
        return AgentResult(
            reply="Mình chưa tìm thấy thông tin cụ thể này trong tài liệu. "
                  "Đã ghi nhận để đội hỗ trợ bổ sung.",
            resolved=False, citations=citations,
        )

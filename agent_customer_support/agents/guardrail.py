from agent_customer_support.agents.prompts import GROUNDING_JUDGE_PROMPT, PROCESS_BLOCK
from agent_customer_support.config import get_settings
from agent_customer_support.llm import complete_structured
from agent_customer_support.llm.schemas import GroundingVerdict

MAX_INPUT_CHARS = 5000


def _sources_block(passages: list[str]) -> str:
    return "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages))


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
            return {
                "pass": False,
                "reason": verdict.reason or "ungrounded",
                "unsupported_claims": verdict.unsupported_claims,
            }
        return {"pass": True, "reason": ""}

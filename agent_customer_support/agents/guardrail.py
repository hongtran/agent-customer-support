from agent_customer_support.agents.prompts import GUARDRAIL_OUTPUT_PROMPT
from agent_customer_support.config import get_settings
from agent_customer_support.llm import complete_structured
from agent_customer_support.llm.schemas import GuardrailVerdict

MAX_INPUT_CHARS = 5000


class GuardrailAgent:
    name = "guardrail"

    async def check_input(self, message: str) -> dict:
        text = (message or "").strip()
        if not text:
            return {"pass": False, "reason": "empty_input"}
        if len(text) > MAX_INPUT_CHARS:
            return {"pass": False, "reason": "oversized_input"}
        return {"pass": True, "reason": ""}

    async def check_output(self, reply: str) -> dict:
        verdict = complete_structured(
            messages=[{"role": "user", "content": reply}],
            system=GUARDRAIL_OUTPUT_PROMPT,
            model=get_settings().model_for("guardrail"),
            schema=GuardrailVerdict,
        )
        # Fail OPEN, deliberately the opposite of triage's fail-safe: a judge that
        # could not answer must not be allowed to silence an already-paid-for reply.
        if verdict is None:
            return {"pass": True, "reason": ""}
        if verdict.flag:
            return {"pass": False, "reason": verdict.reason or "flagged"}
        return {"pass": True, "reason": ""}

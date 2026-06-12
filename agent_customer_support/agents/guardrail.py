import json

from agent_customer_support.agents.prompts import GUARDRAIL_OUTPUT_PROMPT
from agent_customer_support.llm import complete_text

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
        raw = complete_text(
            messages=[{"role": "user", "content": reply}],
            system=GUARDRAIL_OUTPUT_PROMPT,
        )
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {"pass": True, "reason": ""}  # fail-open on parse error
        if data.get("flag"):
            return {"pass": False, "reason": data.get("reason", "flagged")}
        return {"pass": True, "reason": ""}

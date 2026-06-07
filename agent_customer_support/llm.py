from enterprise_llm_service.llm_inference import ai_completion_with_tools
from enterprise_llm_service.llm_inference.llm_inference_base import ai_completion
from agent_customer_support.config import get_settings


def complete_with_tools(*, messages: list[dict], tools: list[dict], system: str | None = None) -> dict:
    return ai_completion_with_tools(
        model=get_settings().agent_model,
        messages=messages,
        tools=tools,
        system=system,
    )


def complete_text(messages: list[dict]) -> str:
    out = ai_completion(model=get_settings().agent_model, messages=messages, max_tokens=1000)
    return out["content"] if isinstance(out, dict) else str(out)

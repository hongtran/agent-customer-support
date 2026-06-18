"""
Clarify / Diagnose behavioral evaluation.

Runs the agent on cases that SHOULD trigger a clarify, a diagnosis, or neither,
and uses an LLM judge to classify the reply's behavior. Validates the policy
fires at the right times (and not when it shouldn't).

Usage:
    # Load .env first:
    set -a && source .env && set +a
    poetry run python scripts/eval_clarify_diagnose.py
"""
import asyncio
import json
import logging
import uuid
from dataclasses import dataclass

logging.basicConfig(level=logging.WARNING, format="%(name)s  %(message)s")
for noisy in ("httpx", "httpcore", "aiobotocore", "aioboto3",
              "aioboto3.resources", "aioboto3.resources.action",
              "aioboto3.resources.factory", "botocore", "botocore.auth",
              "botocore.endpoint", "botocore.parsers", "urllib3",
              "openai", "anthropic", "google"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from agent_customer_support.models import CustomerProfile                       # noqa: E402
from agent_customer_support.stores.customer_registry import CustomerRegistry    # noqa: E402
from agent_customer_support.agents.coordinator import Coordinator               # noqa: E402
from agent_customer_support.llm import complete_with_tools                      # noqa: E402


# ── Test cases ─────────────────────────────────────────────────────────────────
@dataclass
class Case:
    id: str
    turns: list[str]            # one or more user turns (multi-turn for the chain)
    expected: str               # CLARIFY | DIAGNOSE | DIRECT_ANSWER
    note: str


CASES: list[Case] = [
    Case(
        id="pqt_return_branch",
        turns=["Đơn được PQT trả về cho KD thì KD sửa số lượng mẫu trong đơn đã tạo được không?"],
        expected="CLARIFY",
        note="answer forks on order state the agent can't see",
    ),
    Case(
        id="ambiguous_phieu",
        turns=["cách tạo phiếu?"],
        expected="CLARIFY",
        note="'phiếu' = báo giá / PYC / phiếu kết quả",
    ),
    Case(
        id="b7b_date_violation",
        turns=[
            "Em tạo PYC ngày 18/4/2026 nhưng ngày nhận mẫu là 20/4/2026, sửa lại "
            "ngày nhận mẫu cho khớp được không ạ?",
            "Đúng rồi, đây là trường hợp khách đem mẫu tới công ty.",
        ],
        expected="DIAGNOSE",
        note="created PYC before receipt in B7-B = process violation; must point it out",
    ),
    Case(
        id="direct_who_approves",
        turns=["Ai phụ trách bước nghiệm thu hợp đồng?"],
        expected="DIRECT_ANSWER",
        note="unambiguous process question",
    ),
]


# ── Judge prompt ───────────────────────────────────────────────────────────────
JUDGE_SYSTEM = """Bạn phân loại HÀNH VI của câu trả lời Agent (không chấm đúng/sai nội dung).
Nhãn:
- CLARIFY: Agent hỏi lại/xác nhận để lấy thông tin còn thiếu trước khi trả lời.
- DIAGNOSE: Agent chỉ ra user đang làm SAI QUY TRÌNH (nêu điểm sai + lý do) rồi mới hướng dẫn.
- DIRECT_ANSWER: Agent trả lời thẳng, không hỏi lại, không chỉ ra lỗi quy trình.
Trả về JSON: {"verdict":"<nhãn>","reason":"<1 câu>"}"""


def judge(question: str, reply: str) -> dict:
    out = complete_with_tools(
        messages=[{"role": "user", "content": f"Hội thoại:\n{question}\n\nCâu trả lời cuối của Agent:\n{reply}"}],
        tools=[],
        system=JUDGE_SYSTEM,
    )
    raw = (out.get("text") or "").strip()
    # Extract JSON even if wrapped in markdown
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except Exception:
        return {"verdict": "PARSE_ERROR", "reason": raw[:120]}


async def run_eval() -> None:
    reg = CustomerRegistry()
    await reg.init()
    await reg.put(CustomerProfile(
        customer_id="eval_user",
        name="EvalUser",
        enabled_applications=["yeu-cau-thu-nghiem", "lay-mau-quan-trac"],
    ))
    agent = Coordinator()
    passed = 0
    for case in CASES:
        conv_id = f"clarify-{case.id}-{uuid.uuid4().hex[:6]}"
        reply = ""
        joined = ""
        for turn in case.turns:
            joined += f"user: {turn}\n"
            resp = await agent.handle_turn(
                customer_id="eval_user",
                conversation_id=conv_id,
                message=turn,
                attachments=[],
            )
            reply = resp.reply
            joined += f"assistant: {reply}\n"
        verdict = judge(joined, reply)
        ok = verdict.get("verdict") == case.expected
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {case.id}: want {case.expected}, "
              f"got {verdict.get('verdict')} — {verdict.get('reason', '')}")
    print(f"\nScore: {passed}/{len(CASES)}")


if __name__ == "__main__":
    asyncio.run(run_eval())

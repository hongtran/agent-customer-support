"""
Smoke test + interactive debug cho AgentCore.

Usage:
    # Normal (chỉ thấy USER/AGENT)
    poetry run python scripts/smoke_chat.py

    # Debug mode — thấy toàn bộ loop từng bước
    poetry run python scripts/smoke_chat.py --debug

    # Interactive — tự nhập câu hỏi
    poetry run python scripts/smoke_chat.py --interactive

    # Cả hai
    poetry run python scripts/smoke_chat.py --debug --interactive
"""
import asyncio
import logging
import sys

# ── Parse flags trước khi import anything ─────────────────────────────────
DEBUG_MODE       = "--debug"       in sys.argv
INTERACTIVE_MODE = "--interactive" in sys.argv

# ── Setup logging ──────────────────────────────────────────────────────────
level = logging.DEBUG if DEBUG_MODE else logging.WARNING
logging.basicConfig(
    level=level,
    format="%(name)s  %(message)s",
)
# Giữ 3rd-party libs im lặng kể cả khi DEBUG
for noisy in ("httpx", "httpcore", "aiobotocore", "botocore",
              "urllib3", "openai", "anthropic", "google"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from agent_customer_support.models import CustomerProfile              # noqa: E402
from agent_customer_support.stores.customer_registry import CustomerRegistry  # noqa: E402
from agent_customer_support.agent.core import AgentCore                # noqa: E402

# ── Demo question sets ─────────────────────────────────────────────────────
# Chạy mặc định: cả 3 path
DEMO_CASES = {
    "qa": {
        "desc": "PATH A — Q&A (search_knowledge only)",
        "conv": "smoke-qa",
        "msgs": [
            "Tên mẫu khi tạo đơn hàng thì nhập như thế nào?",
            "Địa điểm lấy mẫu không có thì điền gì?",
        ],
    },
    "flow": {
        "desc": "PATH B — Flow guidance (list_flows → get_flow → dẫn từng bước)",
        "conv": "smoke-flow",
        "msgs": [
            # Turn 1: trigger get_flow — yêu cầu hướng dẫn từng bước rõ ràng
            "Hướng dẫn tôi từng bước xử lý PYC sự cố, tôi chưa biết bắt đầu từ đâu",
            # Turn 2: theo flow — agent đang ở step 'tiep_nhan', user xác nhận
            "tôi đã vào menu và thấy phiếu rồi",
            # Turn 3: theo flow — agent ở step 'phe_duyet', user gặp lỗi
            "tôi nhấn phê duyệt nhưng bị lỗi",
        ],
    },
    "feature": {
        "desc": "PATH C — Feature request (log_request)",
        "conv": "smoke-feat",
        "msgs": [
            "Tôi muốn thêm cột địa điểm lấy mẫu vào màn hình danh sách đơn hàng",
        ],
    },
}

SEP  = "─" * 60
SEP2 = "═" * 60


async def chat(agent: AgentCore, customer_id: str, conv_id: str, msg: str) -> None:
    print(f"\n\033[33m▶ USER:\033[0m  {msg}")
    if DEBUG_MODE:
        print(f"\033[90m{SEP}\033[0m")

    reply = await agent.handle_turn(
        customer_id=customer_id,
        conversation_id=conv_id,
        user_msg=msg,
    )

    if DEBUG_MODE:
        print(f"\033[90m{SEP}\033[0m")
    print(f"\033[32m◀ AGENT:\033[0m {reply.reply}")
    if reply.citations:
        print(f"\033[90m  citations: {reply.citations}\033[0m")
    print(f"\033[90m  escalated={reply.escalated}\033[0m")


async def main() -> None:
    print(f"\n\033[1m{SEP2}")
    print("  CenLab Support Agent — Smoke Test")
    mode = ("DEBUG + " if DEBUG_MODE else "") + ("Interactive" if INTERACTIVE_MODE else "Demo")
    print(f"  Mode: {mode}")
    print(f"{SEP2}\033[0m")

    # Seed customer
    reg = CustomerRegistry()
    await reg.init()
    await reg.put(CustomerProfile(
        customer_id="ttp",
        name="TTP",
        enabled_modules=["yeu-cau-thu-nghiem", "xet-nghiem"],
    ))

    agent   = AgentCore()
    conv_id = "smoke1"

    if INTERACTIVE_MODE:
        print("\nNhập câu hỏi (Ctrl+C hoặc 'quit' để thoát):\n")
        try:
            while True:
                try:
                    msg = input("\033[33mBạn:\033[0m ").strip()
                except EOFError:
                    break
                if not msg or msg.lower() in ("quit", "exit", "q"):
                    break
                await chat(agent, "ttp", conv_id, msg)
        except KeyboardInterrupt:
            pass
        print("\nBye!")
    else:
        # Chọn case muốn chạy qua arg: --qa / --flow / --feature / (mặc định: cả 3)
        case_filter = None
        for k in DEMO_CASES:
            if f"--{k}" in sys.argv:
                case_filter = k
                break

        cases_to_run = {case_filter: DEMO_CASES[case_filter]} if case_filter else DEMO_CASES

        for case_key, case in cases_to_run.items():
            print(f"\n\033[1;33m{'─'*60}")
            print(f"  {case['desc']}")
            print(f"{'─'*60}\033[0m")
            for msg in case["msgs"]:
                await chat(agent, "ttp", case["conv"], msg)

        print(f"\n\033[1m{SEP2}\033[0m")


if __name__ == "__main__":
    asyncio.run(main())

"""Evaluate agent accuracy against a CS-authored Excel answer key.

Reads cases straight from an .xlsx with three columns:
    A: Question            — the user message to send to the agent
    B: Applications/modules — (informational) modules the question relates to
    C: Expected Answer     — the ground-truth answer written by the CS team

For each row: run the agent, then ask an LLM judge to compare the agent reply
against the Expected Answer and label it MATCH / PARTIAL / WRONG / REFUSED.
Accuracy is reported as MATCH / total (PARTIAL shown separately as a soft pass).

Usage:
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
    poetry run python scripts/eval_excel_cases.py [path/to/eval_cs.xlsx]
    # default path: ~/Desktop/eval_cs.xlsx
"""
import asyncio, json, logging, os, sys, textwrap, uuid
from pathlib import Path

# Point agent at the KB collection to evaluate BEFORE importing settings.
# Override by exporting PRODUCT_COLLECTION in your env before running.
NEW_COLLECTION = ""
os.environ.setdefault("PRODUCT_COLLECTION", NEW_COLLECTION)

logging.basicConfig(level=logging.WARNING, format="%(name)s  %(message)s")
for n in ("httpx","httpcore","aiobotocore","aioboto3","botocore","urllib3","openai","anthropic","google"):
    logging.getLogger(n).setLevel(logging.WARNING)

import openpyxl
from agent_customer_support.models import CustomerProfile
from agent_customer_support.stores.customer_registry import CustomerRegistry
from agent_customer_support.agents.coordinator import Coordinator
from agent_customer_support.llm import complete_with_tools

C = {"g":"\033[32m","r":"\033[31m","y":"\033[33m","c":"\033[36m","gr":"\033[90m","b":"\033[1m","x":"\033[0m"}

DEFAULT_XLSX = Path.home() / "Desktop" / "eval_cs.xlsx"

# Apps enabled for the eval customer. Broad on purpose so module gating never
# blocks a question; the "Applications/modules" column is kept only for display.
EVAL_APPS = ["Tổ chức - Hệ thống",
    "Quản lý tài liệu",
    "Mua sắm",
    "Phê duyệt phương pháp",
    "Năng lực thử nghiệm",
    "Quản lý khách hàng",
    "Lấy mẫu - Quan trắc",
    "Yêu cầu thử nghiệm",
    "Phòng thí nghiệm",
    "Đo lường - Hiệu chuẩn",
    "Báo cáo thử nghiệm",
    "Quản lý CoA",
    "CV không phù hợp",
    "Thiết bị - Dụng cụ",
    "Hóa chất - Chuẩn",
    "Quản lý kho",
    "Thống kê - Báo cáo"]


def load_cases(path: Path) -> list[dict]:
    """Read (question, modules, expected) rows from the first sheet, skip the header."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.worksheets[0]
    cases = []
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i == 0:
            continue  # header
        question = (str(row[0]).strip() if len(row) > 0 and row[0] else "")
        if not question:
            continue  # skip blank rows
        modules  = (str(row[1]).strip() if len(row) > 1 and row[1] else "")
        expected = (str(row[2]).strip() if len(row) > 2 and row[2] else "")
        cases.append({"stt": len(cases) + 1, "question": question,
                      "modules": modules, "expected": expected})
    return cases


JUDGE_SYS = """Bạn là bộ đánh giá trung lập cho một AI hỗ trợ phần mềm CenLab.
Bạn nhận: câu hỏi người dùng, câu trả lời của Agent, và ĐÁP ÁN CHUẨN do đội CS soạn.
Nhiệm vụ: so sánh câu trả lời của Agent với đáp án chuẩn và chấm nhãn.

Quy tắc chấm:
- MATCH: Agent trả lời tương tự đáp án chuẩn (có thể giải thích thêm), không cần đúng từng chữ.
- PARTIAL: Agent đúng hướng nhưng thiếu bước/ý quan trọng, hoặc chỉ trả lời một phần.
- WRONG: Agent trả lời sai, mâu thuẫn với đáp án chuẩn, hoặc bịa nội dung không có cơ sở.
- REFUSED: Agent từ chối / nói không có trong tài liệu / chỉ ghi nhận yêu cầu, trong khi
  đáp án chuẩn THỰC SỰ có nội dung hướng dẫn (tức là Agent lẽ ra phải trả lời được).

Chỉ dựa vào đáp án chuẩn, KHÔNG dùng kiến thức bên ngoài.
Trả về JSON DUY NHẤT: {"verdict":"<nhãn>","reason":"<1 câu tiếng Việt>"}"""


def judge(case: dict, reply: str) -> dict:
    msg = f"""Câu hỏi: {case['question']}

Câu trả lời của Agent:
{reply}

Đáp án chuẩn của CS:
{case['expected']}

Chấm theo quy tắc."""
    out = complete_with_tools(messages=[{"role":"user","content":msg}], tools=[], system=JUDGE_SYS, model="gpt-4o")
    raw = (out.get("text") or "").strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
    try:
        return json.loads(raw)
    except Exception:
        return {"verdict":"PARSE_ERROR","reason":raw[:100]}


PASS = {"MATCH"}
SOFT = {"PARTIAL"}
VC = {"MATCH":C["g"],"PARTIAL":C["y"],"REFUSED":C["y"],"WRONG":C["r"],"PARSE_ERROR":C["gr"]}


async def main():
    xlsx_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_XLSX
    if not xlsx_path.exists():
        sys.exit(f"{C['r']}Excel not found: {xlsx_path}{C['x']}")
    cases = load_cases(xlsx_path)
    if not cases:
        sys.exit(f"{C['r']}No question rows found in {xlsx_path}{C['x']}")

    reg = CustomerRegistry(); await reg.init()
    await reg.put(CustomerProfile(customer_id="ttp", name="TTP", enabled_applications=EVAL_APPS))
    agent = Coordinator()
    print(f"\n{C['b']}{'═'*74}\n  Agent vs CS answer key — {xlsx_path.name} ({len(cases)} cases)"
          f"\n  collection …{os.environ['PRODUCT_COLLECTION'][-12:]}\n{'═'*74}{C['x']}\n")

    results = []
    for case in cases:
        cid = f"xl-{case['stt']}-{uuid.uuid4().hex[:5]}"
        resp = await agent.handle_turn(customer_id="ttp", conversation_id=cid,
                                       message=case["question"], attachments=[])
        reply = resp.reply
        v = judge(case, reply) if case["expected"] else {"verdict":"NO_REF","reason":"đáp án chuẩn để trống"}
        verdict = v.get("verdict","?"); reason = v.get("reason","")
        results.append({**case, "verdict":verdict, "reply":reply,
                        "escalated":resp.escalated, "reason":reason})
        col = VC.get(verdict, C["gr"])
        print(f"{C['c']}#{case['stt']}{C['x']} {textwrap.shorten(case['question'],70)}")
        print(f"  {col}{verdict}{C['x']}  esc={resp.escalated}  {C['gr']}{reason}{C['x']}")
        print(f"  {C['gr']}Agent: {textwrap.shorten(reply,140)}{C['x']}\n")

    # ── Summary ──────────────────────────────────────────────────────────────
    from collections import Counter
    print(f"{C['b']}{'─'*74}{C['x']}")
    print(f"{C['b']}Verdicts:{C['x']} ", dict(Counter(r['verdict'] for r in results)))
    scored = [r for r in results if r["verdict"] not in ("NO_REF","PARSE_ERROR")]
    p    = sum(1 for r in scored if r["verdict"] in PASS)
    soft = sum(1 for r in scored if r["verdict"] in SOFT)
    if scored:
        acc = p / len(scored) * 100
        sc = C["g"] if acc >= 80 else (C["y"] if acc >= 60 else C["r"])
        print(f"\n{C['b']}Accuracy (MATCH): {sc}{p}/{len(scored)}  ({acc:.0f}%){C['x']}"
              f"  (+{soft} PARTIAL)")
    json.dump(results, open("/tmp/eval_excel_results.json","w"), ensure_ascii=False, indent=2)
    print(f"{C['gr']}Full → /tmp/eval_excel_results.json{C['x']}\n")


if __name__ == "__main__":
    asyncio.run(main())

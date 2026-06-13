"""Evaluate agent vs real CS answers (Excel) on the NEW KB collection.

Cases come from /tmp/excel_cases.json: {stt, content (col F), loai (col G), cs (col H)}.
- "Hướng dẫn sử dụng" → agent SHOULD answer; judge vs CS answer.
- "Nâng cấp"          → feature request; agent SHOULD escalate/log, NOT fabricate.

Usage:
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
    poetry run python scripts/eval_excel_cases.py
"""
import asyncio, json, logging, os, sys, textwrap, uuid

# Point agent at the NEW collection BEFORE importing settings
NEW_COLLECTION = "3fa85f64-5717-4562-b3fc-2c963f66afa7"
os.environ["PRODUCT_COLLECTION"] = NEW_COLLECTION

logging.basicConfig(level=logging.WARNING, format="%(name)s  %(message)s")
for n in ("httpx","httpcore","aiobotocore","aioboto3","botocore","urllib3","openai","anthropic","google"):
    logging.getLogger(n).setLevel(logging.WARNING)

from agent_customer_support.models import CustomerProfile
from agent_customer_support.stores.customer_registry import CustomerRegistry
from agent_customer_support.agents.coordinator import Coordinator
from agent_customer_support.llm import complete_with_tools

C = {"g":"\033[32m","r":"\033[31m","y":"\033[33m","c":"\033[36m","gr":"\033[90m","b":"\033[1m","x":"\033[0m"}

JUDGE_SYS = """Bạn là bộ đánh giá trung lập cho một AI hỗ trợ phần mềm CenLab.
Bạn nhận: câu hỏi người dùng, PHÂN LOẠI yêu cầu, câu trả lời của Agent, và đáp án chuẩn của đội CS (nếu có).

Quy tắc chấm theo PHÂN LOẠI:
- Nếu phân loại = "Hướng dẫn sử dụng":
  - MATCH: Agent trả lời đúng trọng tâm, khớp ý đáp án CS.
  - PARTIAL: Agent đúng một phần / thiếu ý quan trọng.
  - WRONG: Agent trả lời sai hoặc bịa nội dung mâu thuẫn với CS.
  - REFUSED: Agent từ chối / nói không có trong tài liệu / chỉ ghi nhận log (không trả lời được).
- Nếu phân loại = "Nâng cấp" (yêu cầu tính năng mới):
  - CORRECT_ESCALATE: Agent nhận ra đây là đề xuất/tính năng và ghi nhận/chuyển đội hỗ trợ, KHÔNG bịa hướng dẫn.
  - WRONG: Agent bịa ra một hướng dẫn như thể tính năng đã có (hallucinate).

Trả về JSON DUY NHẤT: {"verdict":"<nhãn>","reason":"<1 câu>"}"""

def judge(case, reply):
    cs = case["cs"] or "(đội CS không ghi đáp án)"
    msg = f"""Câu hỏi: {case['content']}
PHÂN LOẠI: {case['loai']}

Câu trả lời của Agent:
{reply}

Đáp án chuẩn của CS:
{cs}

Chấm theo quy tắc."""
    out = complete_with_tools(messages=[{"role":"user","content":msg}], tools=[], system=JUDGE_SYS)
    raw = (out.get("text") or "").strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
    try: return json.loads(raw)
    except Exception: return {"verdict":"PARSE_ERROR","reason":raw[:100]}

PASS = {"MATCH","CORRECT_ESCALATE"}
SOFT = {"PARTIAL"}
VC = {"MATCH":C["g"],"CORRECT_ESCALATE":C["g"],"PARTIAL":C["y"],"REFUSED":C["y"],"WRONG":C["r"],"PARSE_ERROR":C["gr"]}

async def main():
    cases = json.load(open("/tmp/excel_cases.json"))
    reg = CustomerRegistry(); await reg.init()
    await reg.put(CustomerProfile(customer_id="ttp", name="TTP",
                  enabled_modules=["yeu-cau-thu-nghiem","xet-nghiem","quan-trac","cai-dat"]))
    agent = Coordinator()
    print(f"\n{C['b']}{'═'*74}\n  Agent vs CS answers — collection {NEW_COLLECTION[-12:]}\n{'═'*74}{C['x']}\n")

    results=[]
    for case in cases:
        cid=f"xl-{case['stt']}-{uuid.uuid4().hex[:5]}"
        resp = await agent.handle_turn(customer_id="ttp", conversation_id=cid, message=case["content"], attachments=[])
        reply = resp.reply
        has_cs = bool(case["cs"].strip())
        v = judge(case, reply) if has_cs else {"verdict":"NO_CS_REF","reason":"đội CS để trống đáp án"}
        verdict=v.get("verdict","?"); reason=v.get("reason","")
        results.append({**case,"verdict":verdict,"reply":reply,"escalated":resp.escalated,"reason":reason})
        col=VC.get(verdict,C["gr"])
        print(f"{C['c']}#{case['stt']} [{case['loai']}]{C['x']} {textwrap.shorten(case['content'],70)}")
        print(f"  {col}{verdict}{C['x']}  esc={resp.escalated}  {C['gr']}{reason}{C['x']}")
        print(f"  {C['gr']}Agent: {textwrap.shorten(reply,140)}{C['x']}\n")

    # Summary
    from collections import Counter
    print(f"{C['b']}{'─'*74}{C['x']}")
    hd=[r for r in results if r["loai"]=="Hướng dẫn sử dụng"]
    nc=[r for r in results if r["loai"]=="Nâng cấp"]
    print(f"{C['b']}HƯỚNG DẪN SỬ DỤNG ({len(hd)}):{C['x']} ", dict(Counter(r['verdict'] for r in hd)))
    print(f"{C['b']}NÂNG CẤP ({len(nc)}):{C['x']} ", dict(Counter(r['verdict'] for r in nc)))
    scored=[r for r in results if r["verdict"] not in ("NO_CS_REF","PARSE_ERROR")]
    p=sum(1 for r in scored if r["verdict"] in PASS)
    soft=sum(1 for r in scored if r["verdict"] in SOFT)
    print(f"\n{C['b']}PASS (MATCH/CORRECT_ESCALATE): {p}/{len(scored)}{C['x']}  (+{soft} PARTIAL)")
    json.dump(results, open("/tmp/eval_excel_results.json","w"), ensure_ascii=False, indent=2)
    print(f"{C['gr']}Full → /tmp/eval_excel_results.json{C['x']}\n")

if __name__=="__main__":
    asyncio.run(main())

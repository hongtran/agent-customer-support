"""
Hallucination evaluation script.

For each test case:
  1. Run the agent and capture its reply.
  2. Query RAG directly to get the ground-truth passages from the KB.
  3. Send (question + agent-reply + KB passages) to an LLM judge.
  4. Print a verdicts table: GROUNDED | HALLUCINATED | CORRECT_REFUSAL | WRONG_REFUSAL.

Usage:
    # Load .env first:
    set -a && source .env && set +a
    poetry run python scripts/eval_hallucination.py [--debug]
"""
import asyncio
import json
import logging
import os
import sys
import textwrap
import time
import uuid
from dataclasses import dataclass

DEBUG_MODE = "--debug" in sys.argv

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
from agent_customer_support.rag_client import RagClient                         # noqa: E402
from agent_customer_support.config import get_settings                          # noqa: E402
from agent_customer_support.llm import complete_with_tools                      # noqa: E402

# ── Test cases ─────────────────────────────────────────────────────────────────
@dataclass
class EvalCase:
    id: str
    question: str
    # "in_kb"  → agent should answer and the answer must be grounded in KB
    # "not_in_kb" → agent should refuse / log_request, NOT hallucinate
    expected: str
    rag_query: str  # query to fire at RAG to get relevant passages

EVAL_CASES: list[EvalCase] = [
    EvalCase(
        id="pyc_ten_mau",
        question="Tên mẫu khi tạo đơn hàng thì nhập như thế nào?",
        expected="in_kb",
        rag_query="tên mẫu tạo PYC phiếu yêu cầu mẫu",
    ),
    EvalCase(
        id="dia_diem_lay_mau",
        question="Địa điểm lấy mẫu không có thì điền gì?",
        expected="not_in_kb",
        rag_query="địa điểm lấy mẫu trống không có điền gì",
    ),
    EvalCase(
        id="pyc_su_co",
        question="Làm sao xử lý PYC sự cố?",
        expected="in_kb",
        rag_query="xử lý PYC sự cố tiếp nhận phê duyệt",
    ),
    EvalCase(
        id="off_topic_bia",
        question="Cho tôi công thức nấu bia thủ công tại nhà",
        expected="not_in_kb",
        rag_query="công thức nấu bia thủ công",
    ),
    EvalCase(
        id="thiet_bi_khong_co",
        question="Phần mềm có hỗ trợ kết nối máy quang phổ XYZ-9000 đời 2099 không?",
        expected="not_in_kb",
        rag_query="kết nối máy quang phổ thiết bị tích hợp",
    ),
    EvalCase(
        id="huy_pyc",
        question="Làm thế nào để hủy một PYCTN?",
        expected="in_kb",
        rag_query="hủy PYCTN phiếu yêu cầu thử nghiệm",
    ),
    EvalCase(
        id="duyet_ket_qua",
        question="Các bước duyệt kết quả theo phép thử như thế nào?",
        expected="in_kb",
        rag_query="duyệt kết quả theo phép thử các bước",
    ),

    # ── 10 câu hỏi đánh giá mới dựa trên KB cenlab_kb.md ────────────────────

    # Q1 — Quản lý khách hàng: trùng mã
    EvalCase(
        id="khach_hang_ma_trung",
        question="Khi tạo khách hàng mới, những trường nào không được phép trùng nhau?",
        expected="in_kb",
        rag_query="mã khách hàng ID đăng nhập số điện thoại không được trùng tạo mới khách hàng",
    ),

    # Q2 — Báo giá: quy trình duyệt
    EvalCase(
        id="duyet_bao_gia",
        question="Quy trình duyệt báo giá trong CenLab diễn ra như thế nào?",
        expected="in_kb",
        rag_query="duyệt báo giá phê duyệt chờ duyệt đồng ý từ chối",
    ),

    # Q3 — Phiếu kết quả: quy trình duyệt 3 bước
    EvalCase(
        id="duyet_phieu_ket_qua_3_buoc",
        question="Phiếu kết quả (BCTN) cần qua mấy bước duyệt và đó là những bước gì?",
        expected="in_kb",
        rag_query="duyệt phiếu kết quả bước tạo phiếu duyệt trưởng phòng duyệt giám đốc",
    ),

    # Q4 — Tra cứu kết quả: các trạng thái màu sắc
    EvalCase(
        id="trang_thai_ket_qua_mau",
        question="Trạng thái màu vàng trong tra cứu kết quả có nghĩa là gì?",
        expected="in_kb",
        rag_query="tra cứu kết quả trạng thái màu vàng đang nhập kết quả phòng thí nghiệm",
    ),

    # Q5 — Tài liệu nội bộ vs bên ngoài: số bước phân quyền
    EvalCase(
        id="tai_lieu_buoc_phan_quyen",
        question="Tài liệu bên ngoài trong quy trình soạn thảo có bao nhiêu bước phân quyền?",
        expected="in_kb",
        rag_query="tài liệu bên ngoài số bước phân quyền duyệt ban hành phân phối",
    ),

    # Q6 — CV KPH: lưu ý phiếu đã hủy
    EvalCase(
        id="huy_pyctn_khoi_phuc",
        question="Sau khi hủy PYCTN, tôi có thể khôi phục lại phiếu đó không?",
        expected="in_kb",
        rag_query="hủy PYCTN khôi phục lại phiếu đã hủy tab đã hủy",
    ),

    # Q7 — Thiết bị: điều kiện cập nhật từ danh mục
    EvalCase(
        id="cap_nhat_tu_danh_muc_thiet_bi",
        question="Điều kiện nào để có thể cập nhật dữ liệu từ danh mục thiết bị sang thiết bị?",
        expected="in_kb",
        rag_query="cập nhật dữ liệu danh mục thiết bị sang thiết bị dừng sử dụng điều kiện",
    ),

    # Q8 — Ngoài phạm vi: tích hợp ERP
    EvalCase(
        id="off_topic_erp",
        question="CenLab có thể tích hợp với hệ thống ERP SAP của công ty tôi không?",
        expected="not_in_kb",
        rag_query="tích hợp ERP SAP hệ thống bên ngoài kết nối",
    ),

    # Q9 — Ngoài phạm vi: giá dịch vụ cụ thể
    EvalCase(
        id="off_topic_gia_dich_vu",
        question="Giá thử nghiệm chỉ tiêu amoni cho mẫu nước thải là bao nhiêu tiền?",
        expected="not_in_kb",
        rag_query="giá thử nghiệm amoni nước thải bảng giá chi phí",
    ),

    # Q10 — Phân công thử nghiệm: các cách phân công
    EvalCase(
        id="phan_cong_phep_thu",
        question="Trong CenLab có những cách nào để phân công phép thử cho nhân sự?",
        expected="in_kb",
        rag_query="phân công phép thử nhân sự phân công theo hệ thống phân công theo lịch",
    ),
]

# ── Judge prompt ───────────────────────────────────────────────────────────────
JUDGE_SYSTEM = """Bạn là một bộ đánh giá trung lập. Nhiệm vụ: phân loại câu trả lời của Agent.

Nhãn hợp lệ:
- GROUNDED: Agent trả lời và nội dung có cơ sở trực tiếp từ "KB Passages" (không bịa thêm).
- HALLUCINATED: Agent đưa ra thông tin KHÔNG có trong "KB Passages" hoặc bịa thêm chi tiết.
- CORRECT_REFUSAL: Agent từ chối trả lời / ghi nhận log_request vì câu hỏi nằm ngoài tài liệu — đây là hành vi ĐÚNG khi KB không có đáp án.
- WRONG_REFUSAL: Agent từ chối nhưng KB thực ra CÓ đáp án — đây là lỗi bỏ sót.

Trả về JSON duy nhất: {"verdict": "<nhãn>", "reason": "<1-2 câu giải thích ngắn>"}
"""

def judge(question: str, agent_reply: str, kb_passages: list[str]) -> dict:
    passages_text = "\n---\n".join(kb_passages) if kb_passages else "(không có kết quả)"
    user_msg = f"""Câu hỏi: {question}

Câu trả lời của Agent:
{agent_reply}

KB Passages (nội dung tài liệu thực tế từ RAG):
{passages_text}

Phân loại câu trả lời Agent theo nhãn đã định nghĩa."""
    out = complete_with_tools(
        messages=[{"role": "user", "content": user_msg}],
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


# ── Colours ───────────────────────────────────────────────────────────────────
_C = {
    "green":  "\033[32m",
    "red":    "\033[31m",
    "yellow": "\033[33m",
    "cyan":   "\033[36m",
    "grey":   "\033[90m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
}

VERDICT_COLOUR = {
    "GROUNDED":         _C["green"],
    "CORRECT_REFUSAL":  _C["green"],
    "HALLUCINATED":     _C["red"],
    "WRONG_REFUSAL":    _C["yellow"],
    "PARSE_ERROR":      _C["yellow"],
}

VERDICT_PASS = {"GROUNDED", "CORRECT_REFUSAL"}


async def run_eval() -> None:
    SEP2 = "═" * 72
    print(f"\n{_C['bold']}{SEP2}")
    print("  CenLab — Hallucination Evaluation")
    print(f"{SEP2}{_C['reset']}\n")

    # Seed customer
    reg = CustomerRegistry()
    await reg.init()
    await reg.put(CustomerProfile(
        customer_id="eval_user",
        name="EvalUser",
        enabled_modules=["yeu-cau-thu-nghiem", "xet-nghiem"],
    ))

    agent = Coordinator()
    rag   = RagClient()
    settings = get_settings()

    results = []

    for case in EVAL_CASES:
        conv_id = f"eval-{case.id}-{uuid.uuid4().hex[:6]}"
        print(f"{_C['cyan']}▶ [{case.id}]{_C['reset']}  {case.question}")

        # 1. Run agent
        t0 = time.monotonic()
        resp = await agent.handle_turn(
            customer_id="eval_user",
            conversation_id=conv_id,
            message=case.question,
            attachments=[],
        )
        agent_reply = resp.reply
        elapsed_agent = time.monotonic() - t0

        # 2. Query RAG directly for ground truth
        rag_result = await rag.search(
            case.rag_query,
            collection=settings.product_collection,
            top_k=5,
            score_threshold=0.3,
        )
        kb_passages   = rag_result.get("passages", [])
        top_conf      = rag_result.get("top_confidence", 0.0)
        grounding_note = rag_result.get("grounding_note", "")

        # 3. Judge
        t1 = time.monotonic()
        verdict_obj = judge(case.question, agent_reply, kb_passages)
        elapsed_judge = time.monotonic() - t1

        verdict  = verdict_obj.get("verdict", "PARSE_ERROR")
        reason   = verdict_obj.get("reason", "")
        passed   = verdict in VERDICT_PASS
        colour   = VERDICT_COLOUR.get(verdict, _C["grey"])

        results.append({
            "id":           case.id,
            "expected":     case.expected,
            "verdict":      verdict,
            "passed":       passed,
            "agent_reply":  agent_reply,
            "kb_top_conf":  top_conf,
            "reason":       reason,
        })

        # Print result
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {colour}{status}  {verdict}{_C['reset']}")
        print(f"  {_C['grey']}Agent ({elapsed_agent:.1f}s):{_C['reset']} "
              f"{textwrap.shorten(agent_reply, 120)}")
        print(f"  {_C['grey']}RAG top_conf={top_conf:.2f} | {grounding_note[:60]}{_C['reset']}")
        print(f"  {_C['grey']}Judge ({elapsed_judge:.1f}s):{_C['reset']} {reason}")
        print()

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"{_C['bold']}{'─'*72}{_C['reset']}")
    print(f"{'ID':<25}  {'Expected':<12}  {'Verdict':<20}  {'Pass'}")
    print(f"{'─'*25}  {'─'*12}  {'─'*20}  {'─'*4}")
    passed_count = 0
    for r in results:
        colour = VERDICT_COLOUR.get(r["verdict"], _C["grey"])
        tick   = f"{_C['green']}✓{_C['reset']}" if r["passed"] else f"{_C['red']}✗{_C['reset']}"
        if r["passed"]:
            passed_count += 1
        print(
            f"{r['id']:<25}  {r['expected']:<12}  "
            f"{colour}{r['verdict']:<20}{_C['reset']}  {tick}"
        )

    total = len(results)
    score_colour = _C["green"] if passed_count == total else (
        _C["yellow"] if passed_count >= total * 0.7 else _C["red"]
    )
    print(f"\n{_C['bold']}Score: {score_colour}{passed_count}/{total}{_C['reset']}")

    # Save raw results for further inspection
    out_path = "/tmp/eval_hallucination_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"{_C['grey']}Full results → {out_path}{_C['reset']}\n")


if __name__ == "__main__":
    asyncio.run(run_eval())

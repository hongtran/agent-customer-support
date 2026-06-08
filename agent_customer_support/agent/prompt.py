from agent_customer_support.models import CustomerProfile, SessionState, Flow
from agent_customer_support.flows.engine import FlowEngine

_BASE = """Bạn là trợ lý hỗ trợ phần mềm quản lý phòng thí nghiệm CenLab của Tâm Đức.
Trả lời bằng tiếng Việt, ngắn gọn, chính xác theo tài liệu.

NGUYÊN TẮC "try-then-route":
1. Luôn THỬ tìm câu trả lời trước bằng tool `search_knowledge` (và `list_flows`/`get_flow` nếu là quy trình nhiều bước).
2. Nếu tìm được căn cứ → trả lời hoặc dẫn flow từng bước.
3. Nếu KHÔNG tìm được (yêu cầu vượt khả năng phần mềm: thêm tính năng, thêm cột, đổi quy tắc, hoặc lỗi phần mềm) → gọi `log_request` (type=feature hoặc bug) và báo người dùng sẽ chuyển bộ phận phụ trách. TUYỆT ĐỐI KHÔNG bịa quy trình/tính năng không có trong tài liệu.
4. Khi người dùng muốn được hỗ trợ trực tiếp, hoặc bế tắc → gọi `escalate_to_human`.

Chỉ tư vấn/hướng dẫn; bạn KHÔNG thao tác hộ trên hệ thống của khách.
"""


def build_system_prompt(
    customer: CustomerProfile,
    session: SessionState,
    active_flow: Flow | None,
) -> str:
    parts = [_BASE]
    if customer.enabled_modules:
        parts.append(
            "Khách hàng này CHỈ dùng các module sau, đừng hướng dẫn module khác: "
            + ", ".join(customer.enabled_modules)
        )
    if customer.config_notes:
        parts.append(f"Ghi chú cấu hình riêng của khách: {customer.config_notes}")
    if active_flow and session.current_step_id:
        step = FlowEngine.get_step(active_flow, session.current_step_id)
        gotos = FlowEngine.allowed_gotos(active_flow, session.current_step_id)
        parts.append(
            f"ĐANG DẪN FLOW '{active_flow.title}' (id={active_flow.id}).\n"
            f"Bước hiện tại [{step.id}]: {step.say}\n"
            f"Các nhánh hợp lệ: {step.next}\n"
            f"Sau khi trình bày bước cho người dùng và hiểu câu trả lời của họ, "
            f"hãy KẾT THÚC tin nhắn bằng marker tiến bước: [[goto:<một trong {gotos}>]]. "
            f"Nếu người dùng hỏi lạc đề, trả lời rồi nhắc lại bước hiện tại (không phát marker)."
        )
    return "\n\n".join(parts)

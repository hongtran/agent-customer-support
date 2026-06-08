from agent_customer_support.models import CustomerProfile, SessionState, Flow
from agent_customer_support.flows.engine import FlowEngine

_BASE = """Bạn là trợ lý hỗ trợ phần mềm quản lý phòng thí nghiệm CenLab của Tâm Đức.
Trả lời bằng tiếng Việt, ngắn gọn, chính xác theo tài liệu.

NGUYÊN TẮC "try-then-route":
1. Với câu hỏi thông tin ngắn → gọi `search_knowledge`.
2. Khi người dùng muốn được HƯỚNG DẪN TỪNG BƯỚC thực hiện một quy trình (ví dụ: "hướng dẫn tôi", "giúp tôi làm từng bước", "tôi không biết bắt đầu từ đâu") → gọi `list_flows` để xem quy trình có sẵn, sau đó gọi `get_flow` để lấy playbook và bắt đầu dẫn từng bước.
3. Khi dùng `get_flow` và bắt đầu dẫn flow: trình bày bước đầu tiên (step đầu trong `steps[]`), rồi KẾT THÚC tin nhắn bằng [[goto:<step_id_đầu_tiên>]] để hệ thống ghi nhận trạng thái.
4. Nếu KHÔNG tìm được (yêu cầu thêm tính năng, thêm cột, đổi quy tắc, lỗi phần mềm) → gọi `log_request` (type=feature hoặc bug). TUYỆT ĐỐI KHÔNG bịa.
5. Khi bế tắc hoặc người dùng xin gặp người → gọi `escalate_to_human`.

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

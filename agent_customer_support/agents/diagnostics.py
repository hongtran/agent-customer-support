from dataclasses import dataclass


@dataclass(frozen=True)
class DiagnosticRule:
    """A reactive operating-principle rule sourced from the CenLab ops guide.

    `symptom` is a natural-language description shown to the classifier; `guidance`
    is the canonical Vietnamese answer text injected into composition on a match.
    """

    id: str
    symptom: str
    guidance: str


DIAGNOSTIC_RULES: list[DiagnosticRule] = [
    DiagnosticRule(
        id="missing_master_data",
        symptom=(
            "Người dùng không thấy dữ liệu, danh mục/dropdown trống rỗng, "
            "không tìm thấy mục/giá trị để chọn khi thao tác."
        ),
        guidance=(
            "Hãy kiểm tra và chuẩn hoá master data (danh mục nền tảng) trước khi "
            "phát sinh nghiệp vụ — đây là dữ liệu dùng chung cho toàn bộ luồng vận "
            "hành, thiếu master data sẽ khiến các màn hình hiển thị trống."
        ),
    ),
    DiagnosticRule(
        id="no_permission",
        symptom=(
            "Người dùng không có quyền thao tác, menu/chức năng bị ẩn, không thực "
            "hiện được hành động dù đã đăng nhập."
        ),
        guidance=(
            "Hãy liên hệ quản trị hệ thống/admin để được rà soát và phân quyền phù "
            "hợp với mục đích sử dụng và vị trí công việc của bạn."
        ),
    ),
    DiagnosticRule(
        id="ui_not_configured",
        symptom=(
            "Người dùng không thấy cột/trường/thông tin cần xem trên màn hình, "
            "giao diện thiếu thông tin mong đợi."
        ),
        guidance=(
            "Hãy thiết lập lại giao diện theo người dùng: cấu hình hiển thị đúng "
            "các thông tin bạn quan tâm để thao tác nhanh và đúng trọng tâm."
        ),
    ),
    DiagnosticRule(
        id="forgot_lay_mau_button",
        symptom=(
            "Phòng quan trắc (PQT) không nhận được đơn/luồng công việc lấy mẫu; "
            "hồ sơ lấy mẫu bị treo/nghẽn dù Phiếu yêu cầu đã lập."
        ),
        guidance=(
            "Hãy kiểm tra xem PKD đã nhấn nút LẤY MẪU trên Phiếu yêu cầu trước khi "
            "hoàn thành Bước 1 chưa — nếu bỏ qua bước này, phòng quan trắc (PQT) sẽ "
            "không nhận được luồng công việc và hồ sơ bị nghẽn ngay từ đầu."
        ),
    ),
    DiagnosticRule(
        id="nghiem_thu_blocked",
        symptom=(
            "Không tạo/xuất được hồ sơ nghiệm thu hoặc đợt nghiệm thu cho hợp đồng/Phiếu yêu cầu."
        ),
        guidance=(
            "Chỉ có thể tạo đợt/hồ sơ nghiệm thu với các Phiếu yêu cầu ĐÃ XUẤT ĐỦ "
            "KẾT QUẢ. Hãy kiểm tra tiến độ và đảm bảo các Phiếu yêu cầu trong đợt đã "
            "xuất đủ phiếu kết quả (PKQ) trước khi nghiệm thu."
        ),
    ),
    DiagnosticRule(
        id="edit_after_handoff",
        symptom=(
            "Cần sửa đổi/thêm/bớt dữ liệu nhưng không sửa được vì công việc đã "
            "chuyển giao sang bước/ứng dụng khác."
        ),
        guidance=(
            "Tùy trạng thái: nếu dữ liệu còn trong ứng dụng đó, trả về đúng tài "
            "khoản đã tạo/nhập để điều chỉnh; nếu đã chuyển giao nhưng chưa được "
            "tiếp nhận, phủ nhận để trả về ứng dụng trước; nếu đã được tiếp nhận ở "
            "ứng dụng khác, xử lý qua chức năng 'công việc không phù hợp' để điều "
            "chỉnh. Không sửa bằng đường tắt hay trao đổi miệng — phải trả đúng "
            "bước/tài khoản có quyền chỉnh sửa và ghi rõ lý do."
        ),
    ),
]

RULES_BY_ID: dict[str, DiagnosticRule] = {r.id: r for r in DIAGNOSTIC_RULES}

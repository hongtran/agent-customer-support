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
]

RULES_BY_ID: dict[str, DiagnosticRule] = {r.id: r for r in DIAGNOSTIC_RULES}

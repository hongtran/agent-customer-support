"""Canonical application identifiers.

`metadata.application` in Qdrant stores a **slug** — ASCII, lowercase,
underscore-separated (e.g. `lay_mau_quan_trac`). Verified against the live product
collection: all 829 points carry one of 11 such slugs, with no display names and no
nulls.

Everything else in the stack speaks display names. `CustomerProfile.enabled_applications`
holds them, `/widget/customers/{id}/applications` serves them to the UI, and the widget
sends them straight back in `ChatRequest.applications` (e.g. `["Lấy mẫu - Quan trắc"]`).
Feeding those into a Qdrant filter matches nothing, so retrieval must translate before
it filters — that is what `to_slugs` is for.

The map is duplicated from enterprise-llm-service, which owns the ingestion side and is
the source of truth:
`../enterprise-llm-service/enterprise_llm_service/data_processing/extract_info_user_guide.py`
(`_APPLICATION_SLUGS`). It cannot be imported — separate deployable — so keep the two in
sync when applications are added there.
"""

import logging

logger = logging.getLogger(__name__)

# Display name → slug stored in metadata.application.
APPLICATION_SLUGS: dict[str, str] = {
    "Tổ chức - Hệ thống": "to_chuc_he_thong",
    "Quản lý tài liệu": "quan_ly_tai_lieu",
    "Mua sắm": "mua_sam",
    "Phê duyệt phương pháp": "phe_duyet_phuong_phap",
    "Năng lực thử nghiệm": "nang_luc_thu_nghiem",
    "Quản lý khách hàng": "quan_ly_khach_hang",
    "Lấy mẫu - Quan trắc": "lay_mau_quan_trac",
    "Yêu cầu thử nghiệm": "yeu_cau_thu_nghiem",
    "Phòng thí nghiệm": "phong_thi_nghiem",
    "Đo lường - Hiệu chuẩn": "do_luong_hieu_chuan",
    "Báo cáo thử nghiệm": "bao_cao_thu_nghiem",
    "Quản lý CoA": "quan_ly_coa",
    "CV không phù hợp": "cv_khong_phu_hop",
    "Thiết bị - Dụng cụ": "thiet_bi_dung_cu",
    "Hóa chất - Chuẩn": "hoa_chat_chuan",
    "Quản lý kho": "quan_ly_kho",
    "Thống kê - Báo cáo": "thong_ke_bao_cao",
}

# slug → display name, for rendering a slug back to something a human reads.
APPLICATION_NAMES: dict[str, str] = {slug: name for name, slug in APPLICATION_SLUGS.items()}

# Lookup keyed by casefolded display name, so admin free-text entry survives a
# difference in capitalisation.
_BY_FOLDED_NAME: dict[str, str] = {
    name.casefold(): slug for name, slug in APPLICATION_SLUGS.items()
}

_VALID_SLUGS: frozenset[str] = frozenset(APPLICATION_SLUGS.values())


def to_slug(value: str) -> str:
    """Resolve one application identifier to its Qdrant slug.

    Accepts a display name (`"Lấy mẫu - Quan trắc"`), a slug already in canonical
    form (idempotent), or a kebab-case variant (`"yeu-cau-thu-nghiem"`, the
    convention in `seeds/flows`).

    An unrecognised value is returned stripped but otherwise unchanged rather than
    raising: callers scope retrieval with it, and a hard filter on an unknown value
    yields no passages, which the agent already handles. The warning is what makes
    that diagnosable — a silent pass-through here looks identical to "no documents
    for this application".
    """
    cleaned = (value or "").strip()
    if not cleaned:
        return cleaned
    if cleaned in _VALID_SLUGS:
        return cleaned
    mapped = _BY_FOLDED_NAME.get(cleaned.casefold())
    if mapped:
        return mapped
    kebab = cleaned.replace("-", "_")
    if kebab in _VALID_SLUGS:
        return kebab
    logger.warning(
        "unrecognised application %r — passing through unchanged; it will not match "
        "any indexed document. Check APPLICATION_SLUGS against enterprise-llm-service.",
        cleaned,
    )
    return cleaned


def to_slugs(values: list[str] | None) -> list[str] | None:
    """Slug-normalise a list of application identifiers, preserving order.

    Returns None for None/empty so callers can keep using it as "no scope".
    Duplicates collapse — two display names can share a slug only by mistake, but
    the resulting filter should not repeat conditions either way.
    """
    if not values:
        return None
    seen: dict[str, None] = {}
    for v in values:
        slug = to_slug(v)
        if slug:
            seen[slug] = None
    return list(seen) or None

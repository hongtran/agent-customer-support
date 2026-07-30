import logging

from agent_customer_support.applications import (
    APPLICATION_NAMES,
    APPLICATION_SLUGS,
    to_slug,
    to_slugs,
)


def test_display_name_maps_to_slug():
    # the exact payload the widget sends today
    assert to_slug("Lấy mẫu - Quan trắc") == "lay_mau_quan_trac"
    assert to_slug("Phòng thí nghiệm") == "phong_thi_nghiem"


def test_slug_is_idempotent():
    assert to_slug("lay_mau_quan_trac") == "lay_mau_quan_trac"
    assert to_slug(to_slug("Lấy mẫu - Quan trắc")) == "lay_mau_quan_trac"


def test_whitespace_and_case_tolerated():
    assert to_slug("  Lấy mẫu - Quan trắc  ") == "lay_mau_quan_trac"
    assert to_slug("lấy mẫu - quan trắc") == "lay_mau_quan_trac"


def test_kebab_case_variant_resolves():
    # the convention used in seeds/flows
    assert to_slug("yeu-cau-thu-nghiem") == "yeu_cau_thu_nghiem"


def test_unknown_value_passes_through_with_warning(caplog):
    with caplog.at_level(logging.WARNING):
        assert to_slug("Không Tồn Tại") == "Không Tồn Tại"
    assert "unrecognised application" in caplog.text


def test_to_slugs_normalises_list_and_dedupes():
    assert to_slugs(["Lấy mẫu - Quan trắc", "lay_mau_quan_trac"]) == ["lay_mau_quan_trac"]
    assert to_slugs(["Quản lý kho", "Mua sắm"]) == ["quan_ly_kho", "mua_sam"]


def test_to_slugs_empty_is_none():
    # None means "no scope" to the filter builder — empty input must not become a
    # filter that matches nothing.
    assert to_slugs(None) is None
    assert to_slugs([]) is None
    assert to_slugs(["", "   "]) is None


def test_slug_shape_is_filter_safe():
    # slugs go into Qdrant filters and URLs; keep them ASCII/lowercase/underscore
    for slug in APPLICATION_SLUGS.values():
        assert slug.isascii(), slug
        assert slug == slug.lower(), slug
        assert " " not in slug and "-" not in slug, slug


def test_name_map_is_bijective():
    # a duplicated slug would silently merge two applications into one scope
    assert len(APPLICATION_NAMES) == len(APPLICATION_SLUGS)

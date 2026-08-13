from agent_customer_support import doc_images

# One realistic passage shape: a standalone screenshot, then a table row whose first cell
# is a button glyph. Both forms occur throughout the converted guides.
SCREEN_LINE = "![](media/image23.png)"
ICON_ROW = "| ![](media/image24.png) | Chỉnh sửa: Mở form chỉnh sửa bản ghi. |"
SLUG = "cv_khong_phu_hop"


def _meta(application=SLUG):
    return {"application": application}


# ---- rewrite_passages ----


def test_standalone_ref_becomes_a_screen_marker():
    out = doc_images.rewrite_passages(
        [f"Màn hình xử lý:\n{SCREEN_LINE}"], [_meta()], {SLUG: {"image23.png"}}
    )
    assert out[0] == f"Màn hình xử lý:\n[[img:screen:{SLUG}/image23.png]]"


def test_ref_sharing_a_line_becomes_an_icon_marker():
    out = doc_images.rewrite_passages([ICON_ROW], [_meta()], {SLUG: {"image24.png"}})
    assert f"[[img:icon:{SLUG}/image24.png]]" in out[0]
    assert "Chỉnh sửa" in out[0]


def test_marker_is_scoped_by_the_chunks_application():
    """image23.png exists in every guide; the slug is what makes it resolvable."""
    out = doc_images.rewrite_passages(
        [SCREEN_LINE, SCREEN_LINE],
        [_meta("phong_thi_nghiem"), _meta("bao_cao_thu_nghiem")],
        {"phong_thi_nghiem": {"image23.png"}, "bao_cao_thu_nghiem": {"image23.png"}},
    )
    assert out[0] == "[[img:screen:phong_thi_nghiem/image23.png]]"
    assert out[1] == "[[img:screen:bao_cao_thu_nghiem/image23.png]]"


def test_ref_absent_from_the_bucket_is_deleted():
    """Invariant 2: a document whose media was never uploaded shows no images."""
    out = doc_images.rewrite_passages([f"Nhấn {SCREEN_LINE} để tạo."], [_meta()], {SLUG: set()})
    assert "media/" not in out[0]
    assert "img:" not in out[0]
    assert "Nhấn" in out[0] and "để tạo." in out[0]


def test_ref_is_deleted_when_the_chunk_has_no_application():
    """The global-document case in rag_client._build_filter: visible to everyone, but no
    prefix to resolve an image against."""
    out = doc_images.rewrite_passages([SCREEN_LINE], [{}], {SLUG: {"image23.png"}})
    assert out[0] == ""


def test_ref_never_survives_unresolved():
    """Invariant 3. A leaked `media/…` ref would be copied through by the composer and
    render as a broken relative URL in the widget."""
    out = doc_images.rewrite_passages(
        [f"{SCREEN_LINE}\n{ICON_ROW}"], [_meta()], {SLUG: {"image23.png"}}
    )
    assert "media/" not in out[0]  # image24 was unavailable and is gone, not left raw


def test_passage_without_refs_is_returned_untouched():
    passage = "Anh/Chị vui lòng vào menu Nguyên nhân.  Có hai bước."
    out = doc_images.rewrite_passages([passage], [_meta()], {})
    assert out[0] == passage


def test_metas_shorter_than_passages_does_not_raise():
    out = doc_images.rewrite_passages([SCREEN_LINE], [], {SLUG: {"image23.png"}})
    assert out == [""]


# ---- slugs_with_refs ----


def test_slugs_with_refs_only_lists_documents_that_could_use_an_image():
    slugs = doc_images.slugs_with_refs(
        [SCREEN_LINE, "không có hình", ICON_ROW],
        [_meta("phong_thi_nghiem"), _meta("quan_ly_kho"), _meta()],
    )
    assert slugs == {"phong_thi_nghiem", SLUG}


# ---- select ----


def _screen(n: int) -> str:
    return f"[[img:screen:{SLUG}/image{n}.png]]"


def _icon(n: int) -> str:
    return f"[[img:icon:{SLUG}/image{n}.png]]"


# The catalog every `select` call is checked against: what this turn's passages offered.
CATALOG = {SLUG: {f"image{n}.png" for n in list(range(1, 10)) + [20, 21, 24]}}


def test_select_keeps_a_valid_marker_in_place():
    text = f"Nhấn {_icon(24)} để tạo hồ sơ."
    assert doc_images.select(text, CATALOG) == text


def test_select_drops_a_wellformed_marker_naming_a_file_that_was_never_offered():
    """The catalog is the whitelist, not the regex. A model that invents an image number
    under a real slug writes a perfectly well-formed marker; signing a URL for it would
    render a broken image."""
    out = doc_images.select(f"Xem hình. [[img:screen:{SLUG}/image999.png]]", CATALOG)
    assert "img:" not in out
    assert out == "Xem hình."


def test_select_drops_a_marker_for_an_application_not_in_this_turn():
    out = doc_images.select("Xem [[img:screen:quan_ly_kho/image1.png]] nhé.", CATALOG)
    assert "img:" not in out


def test_select_drops_a_malformed_marker():
    out = doc_images.select("Xem [[img:image9.png]] nhé.", CATALOG)
    assert "[[img" not in out
    assert "Xem" in out and "nhé." in out


def test_select_dedupes_keeping_the_first_position():
    out = doc_images.select(f"A {_screen(1)} B {_screen(1)} C", CATALOG)
    assert out.count(_screen(1)) == 1
    assert out.index("B") > out.index(_screen(1))


def test_select_caps_and_prefers_screenshots():
    icons = " ".join(_icon(n) for n in range(1, 7))
    text = f"{icons} {_screen(20)} {_screen(21)}"
    out = doc_images.select(text, CATALOG, max_images=5)

    assert _screen(20) in out and _screen(21) in out  # both screens survive the cap
    kept_icons = [n for n in range(1, 7) if _icon(n) in out]
    assert kept_icons == [1, 2, 3]  # remaining budget goes to icons in document order


def test_select_with_zero_budget_removes_every_marker():
    out = doc_images.select(f"Nhấn {_icon(24)} để tạo.", CATALOG, max_images=0)
    assert "img:" not in out
    assert out == "Nhấn để tạo."


def test_select_leaves_marker_free_text_byte_identical():
    """The headline requirement: an answer from a document with no images must be exactly
    what the composer wrote, including its own spacing."""
    text = "Anh/Chị vui lòng chọn:\n\n- Phiếu yêu cầu  thử nghiệm\n- Mẫu PTN\n"
    assert doc_images.select(text, CATALOG) == text


def test_select_tidies_the_line_a_removed_screenshot_left_behind():
    out = doc_images.select(f"Bước 1\n{_screen(9)}\nBước 2", CATALOG, max_images=0)
    assert out == "Bước 1\nBước 2"


# ---- markers_in / presign_markers ----


def test_markers_in_dedupes_and_preserves_order():
    text = f"{_screen(1)} {_icon(2)} {_screen(1)}"
    assert doc_images.markers_in(text) == [
        ("screen", SLUG, "image1.png"),
        ("icon", SLUG, "image2.png"),
    ]


def test_presign_markers_emits_markdown_with_kind_in_the_alt():
    out = doc_images.presign_markers(
        f"Nhấn {_icon(24)} để tạo.", lambda slug, name: f"https://signed/{slug}/{name}"
    )
    assert out == f"Nhấn ![icon](https://signed/{SLUG}/image24.png) để tạo."


def test_presign_markers_drops_an_image_it_cannot_sign():
    def boom(slug, name):
        raise RuntimeError("no s3")

    out = doc_images.presign_markers(f"Nhấn {_icon(24)} để tạo.", boom)
    assert out == "Nhấn để tạo."


def test_presign_markers_leaves_marker_free_text_untouched():
    text = "Vào  menu X.\n\nRồi bấm Lưu."
    assert doc_images.presign_markers(text, lambda s, n: "x") == text


def test_strip_removes_markers_and_leaves_other_text_alone():
    assert doc_images.strip(f"A {_screen(1)} B") == "A B"
    assert doc_images.strip("A  B") == "A  B"


# ---- key layout ----


def test_s3_key_is_derived_from_the_slug():
    assert doc_images.s3_key(SLUG, "image23.png") == f"doc_images/{SLUG}/image23.png"

"""The citation whitelist: what may be shown as a source, and what may not.

Mirrors tests/test_doc_images.py — same guarantee, different payload. A composed answer
declares its sources and the section of each it drew from; only what this turn actually
offered survives, and no filename ever appears.
"""

from agent_customer_support import citations as cite
from agent_customer_support.citations import PROCESS_DOC_ID
from agent_customer_support.llm.schemas import CitedSource

# Shapes taken from the real corpus (eval/results/retrieval-*.csv).
PYC_HEADING = "Quy trình tổng thể – Vòng đời PYC"
PYC_PASSAGE = (
    "Quy trình tổng thể và vòng đời của một Phiếu Yêu Cầu (PYC)\n\n"
    "Chunk này mô tả quy trình tổng thể hay vòng đời của một PYC.\n\n"
    "| Thanh lý mẫu | Xử lý/tiêu huỷ mẫu sau khi hoàn tất thử nghiệm |\n\n"
    f"### 2.1. {PYC_HEADING}\n\n"
    "Vòng đời PYC không được quy định cố định theo một số bước chung."
)
IMPORT_HEADING = "Import ký hiệu mẫu"
# The reported failure: a chunk whose first line is a prepended summary sentence, with
# the real headings further down carrying bold markers and trailing colons.
IMPORT_PASSAGE = (
    "Quản lý, nhập ký hiệu và gán phép thử cho mẫu trong PYC\n\n"
    "Chunk này mô tả các thao tác quản lý mẫu đã tạo như sửa, xóa, sao chép mẫu.\n\n"
    "##### Thao tác với mẫu đã tạo\n\n"
    "- Sửa mẫu: chọn dòng và nhấn biểu tượng bút chì.\n\n"
    f"##### **{IMPORT_HEADING}:**\n\n"
    "1. Mở chức năng Import ký hiệu mẫu tại PYC đang thao tác.\n\n"
    "**Điều kiện áp dụng:** PYC chưa khai báo ký hiệu biên bản.\n\n"
    "##### Điều kiện và phân quyền khi Xóa/Hủy mẫu hoặc phép thử:\n\n"
    "- CENLAB phân quyền theo vai trò.\n\n"
    "##### **Cách khai báo mẫu gộp**:\n\n"
    "1. Khai báo đầy đủ các mẫu đơn trước."
)

NO_HEADING_PASSAGE = (
    "| Vai trò | Không | Gán một hoặc nhiều vai trò (multi-select). |\n"
    "| Giới tính | Không | Chọn từ danh sách. |"
)


def _meta(doc_id="doc-9", application="yeu_cau_thu_nghiem", **kw):
    return {"doc_id": doc_id, "application": application, **kw}


def _cite(id, section=""):
    return CitedSource(id=id, section=section)


# ---- sections(): the whitelist, never the output ----


def test_numbered_heading_loses_its_number_and_keeps_everything_else():
    """The number is an artefact of the document outline. Diacritics and the en-dash are
    the actual title and must survive verbatim."""
    assert cite.sections(PYC_PASSAGE) == [PYC_HEADING]


def test_multi_level_numbering_and_curly_quotes():
    passage = "## 3. Menu “Cơ cấu tổ chức”\n\nbody\n\n### 4.2.3. Tạo mới biên bản\n\nbody"
    assert cite.sections(passage) == ["Menu “Cơ cấu tổ chức”", "Tạo mới biên bản"]


def test_a_document_title_and_its_subsection_are_both_candidates():
    """A chunk can open with the guide's own `#` title before the `##` section it really
    covers. Which one applies is not decidable here — the composer picks, and both must
    be available for it to pick from."""
    passage = "# Hướng dẫn sử dụng - Tổ chức - Hệ thống\n\n## Cài đặt phân quyền\n\nCác cột..."
    assert cite.sections(passage) == [
        "Hướng dẫn sử dụng - Tổ chức - Hệ thống",
        "Cài đặt phân quyền",
    ]


def test_a_chunk_that_opens_mid_table_offers_nothing():
    """Normal outcome, not an error — the corpus contains these."""
    assert cite.sections(NO_HEADING_PASSAGE) == []


def test_bold_text_and_inline_hashes_are_not_headings():
    passage = "**The menu consists of 4 tabs:**\n\nxem mục #3 để biết thêm\n| a # b | c |"
    assert cite.sections(passage) == []


def test_an_image_marker_on_the_heading_line_is_not_part_of_the_title():
    """The guides put the section's screenshot on the heading line itself — sharing a
    line is precisely what makes doc_images call it an icon — and `_with_images` rewrites
    the ref before compose sees the passage, so the marker lands inside the captured
    heading text. The colon it leaves stranded comes off with it."""
    raw = "##### **Import ký hiệu mẫu:** [[img:icon:yeu_cau_thu_nghiem/image20.png]]"
    assert cite.sections(raw) == ["Import ký hiệu mẫu"]


def test_a_heading_that_is_only_an_image_offers_nothing():
    assert cite.sections("##### [[img:screen:a/image1.png]]") == []


def test_a_colon_inside_a_real_title_survives():
    """Only the marker is removed, never everything after the colon — a title can
    legitimately contain one, and truncating there would quietly lose half of it."""
    assert cite.sections("### 3.3. Tab - Vai trò: quyền truy cập") == [
        "Tab - Vai trò: quyền truy cập"
    ]


def test_headings_are_cleaned_of_markers_numbers_and_trailing_colons():
    """Every row here is transcribed from the corpus, so this is the actual contract.

    The guides put emphasis and a trailing colon INSIDE the heading text, and the reader
    wants none of it.
    """
    cases = {
        "##### **Import ký hiệu mẫu:**": "Import ký hiệu mẫu",
        "##### **Cách khai báo mẫu gộp**:": "Cách khai báo mẫu gộp",
        "##### Điều kiện và phân quyền khi Xóa/Hủy mẫu:": "Điều kiện và phân quyền khi Xóa/Hủy mẫu",
        "### 2.1. Quy trình tổng thể – Vòng đời PYC": "Quy trình tổng thể – Vòng đời PYC",
        "## 3. Menu “Cơ cấu tổ chức”": "Menu “Cơ cấu tổ chức”",
        "### 4.2.3. Tạo mới biên bản": "Tạo mới biên bản",
        "### 3.3. Tab - Vai trò": "Tab - Vai trò",
    }
    for raw, want in cases.items():
        assert cite.sections(raw) == [want], raw


def test_an_interior_asterisk_survives_cleaning():
    """`\\*` is pandoc's escape for a required-field marker, and it is content. Only the
    paired emphasis markers and the edges come off — a blanket strip("*") would eat it."""
    assert cite.sections("### Khai báo Tên mẫu gộp \\* và Số lượng") == [
        "Khai báo Tên mẫu gộp * và Số lượng"
    ]


def test_the_reported_chunk_offers_its_four_real_sections():
    assert cite.sections(IMPORT_PASSAGE) == [
        "Thao tác với mẫu đã tạo",
        IMPORT_HEADING,
        "Điều kiện và phân quyền khi Xóa/Hủy mẫu hoặc phép thử",
        "Cách khai báo mẫu gộp",
    ]


# ---- catalog ----


def test_catalog_indexes_passages_positionally():
    cat = cite.catalog([_meta(), _meta(doc_id="doc-2", application="mua_sam")])
    assert cat["0"].doc_id == "doc-9"
    assert cat["1"].application == "Mua sắm"  # slug rendered as display name
    assert cat["0"].kind == "guide"


def test_a_guide_falls_back_to_its_application_before_any_section_is_resolved():
    assert cite.catalog([_meta()])["0"].label == "Yêu cầu thử nghiệm"


def test_a_global_chunk_with_no_application_gets_the_generic_label():
    """Untagged documents are global by design (rag_client._build_filter). With no module
    and no heading there is nothing specific left to say — but still no filename."""
    cat = cite.catalog([_meta(application=None)])
    assert cat["0"].application is None
    assert cat["0"].label == "Tài liệu hướng dẫn CenLab"


def test_process_is_always_declarable():
    """Kept in the catalog even though `select` never shows it: without this id the
    model would attribute a process claim to whichever passage is nearest, which is the
    fabricated citation this module exists to prevent."""
    cat = cite.catalog([])
    assert cat[PROCESS_DOC_ID].kind == "process"
    assert cat[PROCESS_DOC_ID].label == "Quy trình vận hành CenLab"


def test_a_chunk_with_no_doc_id_is_not_citable():
    """Nothing to attribute it to and nothing to dedupe on, so declaring it fails the
    whitelist check rather than producing a row."""
    assert "0" not in cite.catalog([{"application": "mua_sam"}])


def test_qa_records_are_numbered_separately_and_namespaced():
    cat = cite.catalog([_meta()], [{"source_doc_id": "abc", "confidence": 0.9}])
    assert cat["qa:0"].doc_id == "qa:abc"  # cannot collide with a guide UUID
    assert cat["qa:0"].label == "Đáp án đã được CS xác nhận"
    assert cat["0"].doc_id == "doc-9"  # product numbering is untouched


# ---- select: sections ----


def test_the_declared_section_becomes_the_label():
    cat = cite.catalog([_meta()])
    got = cite.select([_cite("0", PYC_HEADING)], cat, [PYC_PASSAGE])
    assert len(got) == 1
    assert got[0].section == PYC_HEADING
    assert got[0].label == PYC_HEADING
    assert got[0].application == "Yêu cầu thử nghiệm"


def test_the_model_may_re_add_the_hashes_or_the_number():
    """A transcription slip must not cost the user a real section."""
    cat = cite.catalog([_meta()])
    for declared in (f"### 2.1. {PYC_HEADING}", f"2.1. {PYC_HEADING}", f"  {PYC_HEADING}  "):
        got = cite.select([_cite("0", declared)], cat, [PYC_PASSAGE])
        assert got[0].section == PYC_HEADING, declared


def test_the_clean_heading_matches_the_marked_up_one_in_the_passage():
    """The regression behind the reported bug. The prompt asks for `Import ký hiệu mẫu`;
    the passage spells it `##### **Import ký hiệu mẫu:**`. Comparing the raw forms
    rejected exactly the well-behaved answers, so a correct declaration silently lost its
    section."""
    cat = cite.catalog([_meta()])
    got = cite.select([_cite("0", IMPORT_HEADING)], cat, [IMPORT_PASSAGE])
    assert got[0].section == IMPORT_HEADING
    assert got[0].label == IMPORT_HEADING


def test_the_model_may_keep_the_colon_or_the_bold_it_was_told_to_drop():
    """Both sides are cleaned, so residual drift still matches rather than costing the
    user a real section."""
    cat = cite.catalog([_meta()])
    for declared in ("**Import ký hiệu mẫu:**", "Import ký hiệu mẫu:", "##### Import ký hiệu mẫu"):
        got = cite.select([_cite("0", declared)], cat, [IMPORT_PASSAGE])
        assert got[0].section == IMPORT_HEADING, declared


def test_the_prepended_summary_line_is_not_a_section():
    """What the composer actually reached for. It is the chunk's first line and the most
    title-shaped text in it, but it is not a heading — so it is rejected, and the row
    falls back to the module rather than showing a summary sentence as a title."""
    cat = cite.catalog([_meta()])
    got = cite.select(
        [_cite("0", "Quản lý, nhập ký hiệu và gán phép thử cho mẫu trong PYC")],
        cat,
        [IMPORT_PASSAGE],
    )
    assert got[0].section is None
    assert got[0].label == "Yêu cầu thử nghiệm"


def test_an_invented_section_is_dropped_and_the_citation_survives():
    """Pointing at the wrong part of a guide is worse than pointing at the guide. Same
    failure class as an invented image number, one level down."""
    cat = cite.catalog([_meta()])
    got = cite.select([_cite("0", "Mục không có thật")], cat, [PYC_PASSAGE])
    assert got[0].section is None
    assert got[0].label == "Yêu cầu thử nghiệm"


def test_a_heading_less_chunk_cites_as_its_application():
    cat = cite.catalog([_meta()])
    got = cite.select([_cite("0")], cat, [NO_HEADING_PASSAGE])
    assert got[0].section is None
    assert got[0].label == "Yêu cầu thử nghiệm"


def test_only_the_used_heading_appears_not_every_heading_in_the_chunk():
    """`sections` is a whitelist, never output. A chunk offering two headings must not
    report both — that would claim the answer used material it did not."""
    passage = "## Cài đặt phân quyền\n\nbody\n\n## Cơ cấu tổ chức\n\nbody"
    cat = cite.catalog([_meta()])
    got = cite.select([_cite("0", "Cơ cấu tổ chức")], cat, [passage])
    assert [c.section for c in got] == ["Cơ cấu tổ chức"]


# ---- select: ids and dedupe ----


def test_declared_sources_become_citations_in_order():
    cat = cite.catalog([_meta(), _meta(doc_id="doc-2", application="mua_sam")])
    got = cite.select([_cite("1"), _cite("0")], cat, ["p0", "p1"])
    assert [c.doc_id for c in got] == ["doc-2", "doc-9"]


def test_an_invented_index_is_dropped():
    """Six passages came back and the model declared the seventh. A source row for it
    would be a fabricated citation — the worst failure for a feature whose whole point
    is letting the user check."""
    assert cite.select([_cite("7")], cite.catalog([_meta()]), ["p0"]) == []


def test_a_wellformed_id_outside_the_catalog_is_dropped():
    """Shape is not the guard. `qa:3` is perfectly well-formed; it just was not offered
    this turn, and that is the only thing that decides."""
    cat = cite.catalog([_meta()], [{"source_doc_id": "abc"}])
    assert cite.select([_cite("qa:3")], cat, ["p0"], ["cs"]) == []


def test_one_chunk_cited_for_two_sections_is_two_rows():
    """The whole point of sections: two parts of one guide are two different places for
    the user to look."""
    passage = "## Cài đặt phân quyền\n\nbody\n\n## Cơ cấu tổ chức\n\nbody"
    cat = cite.catalog([_meta()])
    got = cite.select(
        [_cite("0", "Cài đặt phân quyền"), _cite("0", "Cơ cấu tổ chức")], cat, [passage]
    )
    assert [c.section for c in got] == ["Cài đặt phân quyền", "Cơ cấu tổ chức"]


def test_the_same_section_twice_is_one_row():
    cat = cite.catalog([_meta(), _meta()])
    got = cite.select(
        [_cite("0", PYC_HEADING), _cite("1", PYC_HEADING)], cat, [PYC_PASSAGE, PYC_PASSAGE]
    )
    assert len(got) == 1


def test_two_heading_less_chunks_of_one_application_collapse():
    """They would render as the same line twice, and a duplicated row reads as a bug."""
    cat = cite.catalog([_meta(), _meta(doc_id="doc-2")])
    got = cite.select([_cite("0"), _cite("1")], cat, [NO_HEADING_PASSAGE] * 2)
    assert [c.label for c in got] == ["Yêu cầu thử nghiệm"]


def test_common_id_formatting_slips_still_resolve():
    cat = cite.catalog([_meta()])
    for raw in ("[0]", " 0 ", "passage 0", "Đoạn trích 0"):
        assert [c.doc_id for c in cite.select([_cite(raw)], cat, ["p0"])] == ["doc-9"], raw


def test_the_process_is_declarable_but_never_displayed():
    """It is our system prompt, not a document the customer has. Naming it answers
    "where did this come from?" with somewhere they cannot go."""
    assert PROCESS_DOC_ID in cite.catalog([])  # declarable
    assert cite.select([_cite(PROCESS_DOC_ID)], cite.catalog([]), []) == []  # not shown


def test_a_qa_record_is_declarable_but_never_displayed():
    """Same reason: the CS-verified Q&A store is ours, not theirs."""
    cat = cite.catalog([], [{"source_doc_id": "abc"}])
    assert "qa:0" in cat
    assert cite.select([_cite("qa:0")], cat, [], ["cs answer"]) == []


def test_an_answer_grounded_only_in_internal_sources_shows_nothing():
    """The honest outcome — no row at all, rather than a source the reader cannot open.
    The widget renders no `Nguồn` block for an empty list."""
    cat = cite.catalog([_meta()], [{"source_doc_id": "abc"}])
    got = cite.select([_cite(PROCESS_DOC_ID), _cite("qa:0")], cat, [PYC_PASSAGE], ["cs"])
    assert got == []


def test_a_mixed_answer_shows_only_the_guide():
    cat = cite.catalog([_meta()], [{"source_doc_id": "abc"}])
    got = cite.select(
        [_cite(PROCESS_DOC_ID), _cite("0", PYC_HEADING), _cite("qa:0")],
        cat,
        [PYC_PASSAGE],
        ["cs"],
    )
    assert [c.label for c in got] == [PYC_HEADING]


def test_nothing_declared_means_nothing_cited():
    cat = cite.catalog([_meta()])
    assert cite.select([], cat, ["p0"]) == []
    assert cite.select(None, cat, ["p0"]) == []


# ---- no filename, anywhere ----


def test_a_citation_has_nowhere_to_put_a_filename():
    """The constraint, asserted directly: the source document's name is internal, and
    this type exists to be shown to the customer."""
    from agent_customer_support.models import Citation

    assert "title" not in Citation.model_fields
    cat = cite.catalog([_meta(url="chunks/HDSD-YCTN.md-chunk_02.pdf")])
    dumped = str(cite.select([_cite("0", PYC_HEADING)], cat, [PYC_PASSAGE])[0].model_dump())
    assert "HDSD" not in dumped and ".md" not in dumped


# ---- passages_for ----


def test_cited_passage_text_is_returned_for_the_judge():
    cat = cite.catalog([_meta(), _meta(doc_id="doc-2")], [{"source_doc_id": "abc"}])
    got = cite.passages_for([_cite("1"), _cite("qa:0")], cat, ["p0", "p1"], ["cs answer"])
    assert got == ["p1", "cs answer"]


def test_one_chunk_cited_twice_is_handed_to_the_judge_once():
    """Two sections, one passage. Sending the same text twice would only waste tokens."""
    cat = cite.catalog([_meta()])
    got = cite.passages_for([_cite("0", "A"), _cite("0", "B")], cat, ["p0"])
    assert got == ["p0"]


def test_the_process_contributes_no_passage():
    """It is not a passage. The judge gets the process block in its own system prefix,
    the same way the composer did."""
    cat = cite.catalog([_meta()])
    assert cite.passages_for([_cite(PROCESS_DOC_ID), _cite("0")], cat, ["p0"]) == ["p0"]


def test_an_index_past_the_passage_list_is_skipped_not_raised():
    """Defensive: the catalog and the passage list are built from the same metas, so
    this should not happen — but a mid-turn IndexError would cost an answer that was
    already paid for."""
    assert cite.passages_for([_cite("0")], cite.catalog([_meta()]), []) == []

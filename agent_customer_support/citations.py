"""Which sources an answer used: declared ids → validated, readable citations.

The composer returns the sources it used as a list of `(id, section)` declarations
alongside its prose (see `llm.schemas.ComposedAnswer`). This module is the whole transform
between that raw declaration and the citation list the widget shows. It is pure — no I/O,
no Qdrant — because validating a declaration needs nothing the turn does not already hold:
the passages and their metadata came back from `RagClient.search` before compose was ever
called.

Three invariants decide what the user sees, and nothing else does:

1. **The catalog, not the shape, is the guard.** `select` keeps an id only if this turn's
   catalog holds it. A model that declares `"7"` when six passages came back writes a
   perfectly well-formed id, and showing a source for it would be a fabricated citation —
   the worst possible failure for a feature whose entire purpose is letting the user
   check. Same rule, same reason as `doc_images.select`.

2. **A section is never invented either.** The heading the composer names is matched
   against the headings actually present in that passage (`sections`). No match drops the
   section and keeps the citation: a source pointing at the wrong part of a guide is worse
   than one pointing at the guide. `sections` is a whitelist, never output — the headings
   a chunk merely *offers* must not be listed as though the answer used them.

3. **Only a source the customer can actually open is shown.** A row must answer "where
   did this come from?" with somewhere the reader can go and check. That rules out two
   things. Filenames: internal, so `Citation` has no field one could sit in and this
   module holds none to put there — the constraint is structural, not a convention. And
   the two non-document sources: the operating-process block and the CS-verified Q&A
   store are ours, not the customer's, so citing them names something unresolvable. An
   answer grounded only in those simply shows no sources, which is the honest outcome.

Three kinds of source exist, and the distinction is between what may be DECLARED and what
may be DISPLAYED. All three are declarable — product guides by position (`"0"`, `"1"`, …),
CS-verified Q&A records by `"qa:<i>"`, and the always-on process block by the pseudo-id
`quy_trinh_chung`. Only guides are displayable, and only guides carry a section; `select`
is the single place that filter is applied, and its docstring explains why the other two
must remain declarable rather than being removed from the prompt.
"""

import re

from agent_customer_support import doc_images
from agent_customer_support.applications import APPLICATION_NAMES
from agent_customer_support.llm.schemas import CitedSource
from agent_customer_support.models import Citation

# The process block (agents/prompts.PROCESS_CONTEXT) is a system prefix, not a retrieved
# document. It still needs an id the composer can name and a label the user can read.
PROCESS_DOC_ID = "quy_trinh_chung"
PROCESS_LABEL = "Quy trình vận hành CenLab"
QA_LABEL = "Đáp án đã được CS xác nhận"

# A guide chunk with no heading AND no application. Untagged documents are global by
# design (rag_client._build_filter), so this is reachable, and it is the only case with
# nothing specific left to say.
GENERIC_GUIDE_LABEL = "Tài liệu hướng dẫn CenLab"

PROCESS_CITATION = Citation(
    doc_id=PROCESS_DOC_ID,
    label=PROCESS_LABEL,
    section=None,
    application=None,
    kind="process",
    confidence=1.0,
)

_QA_PREFIX = "qa:"

# A markdown ATX heading on its own line. The guides use levels 1-3 in practice; matching
# up to 6 costs nothing. Anchored per line via MULTILINE so a `#` inside a table cell or
# mid-sentence is not a heading.
_HEADING_RE = re.compile(r"^[ \t]{0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$", re.MULTILINE)

# Leading section numbering: "1.", "3.2.", "4.2.3." — verified against the corpus. Dropped
# from the display text because the number is an artefact of the document's outline, and
# the composer is told to drop it too, so both sides of the match agree.
_LEAD_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)*\.?[ \t]*")

# Markdown emphasis and pandoc's backslash escapes, which the guides put INSIDE heading
# text: `##### **Import ký hiệu mẫu:**`. Removed globally because `**` is never content.
_EMPHASIS_RE = re.compile(r"(\*\*|__|\\)")


def _clean_heading(raw: str) -> str:
    r"""Heading text as a person would write it: no markers, no outline number, no colon.

    Runs on BOTH the extracted candidate and the composer's declaration, which is the
    whole point. The prompt asks for the clean form, so a matcher comparing against the
    raw form would reject exactly the well-behaved answers — `**Import ký hiệu mẫu:**`
    never equals the `Import ký hiệu mẫu` it asked for.

    An image marker can sit ON the heading line — `##### **Import ký hiệu mẫu:**
    ![](media/image20.png)` is real, and sharing a line is exactly what makes
    `doc_images._kind_for` call it an icon. By the time a passage reaches here the ref has
    been rewritten to `[[img:icon:…]]`, so the marker is stripped first, using
    doc_images' own regex rather than a second copy of it. Raw `media/…` refs need no
    handling: `rewrite_passages` guarantees none survives, rewritten or deleted.

    Only the marker is removed, not everything after the colon. A colon is legitimate
    inside a title (`Tab - Vai trò: quyền truy cập`), so truncating there would quietly
    lose half of one; the trailing colon left behind by the marker's removal comes off
    with the normal rstrip below.

    `*` and `_` are stripped only at the EDGES after the paired markers are gone. A
    heading like `Khai báo Tên mẫu gộp \* và Số lượng` carries a real asterisk mid-string
    (the required-field marker), and a blanket strip would eat it.
    """
    text = _LEAD_NUMBER_RE.sub("", doc_images.strip(raw or "").strip())
    text = _EMPHASIS_RE.sub("", text)
    return text.strip(" \t*_").rstrip(":：").strip()


def sections(passage: str) -> list[str]:
    """The headings a passage offers, in document order, deduped.

    Used ONLY as the whitelist for a declared section — never rendered. Listing every
    heading a chunk contains would say "the answer used all of these", which is exactly
    the claim we cannot make.

    Returns [] for a passage with no heading. That is a normal outcome, not an error: the
    corpus contains chunks that open mid-table, and they simply cite without a section.
    """
    out: list[str] = []
    seen: set[str] = set()
    for m in _HEADING_RE.finditer(passage or ""):
        text = _clean_heading(m.group(2))
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _normalize_section(text: str) -> str:
    """Comparison key for a heading.

    Tolerates the model re-typing the `#` markers, keeping the outline number or the
    trailing colon it was told to drop, re-adding bold, or any whitespace difference.
    Diacritics and case-sensitive spelling are NOT flattened — two sibling sections can
    differ only by a word, and collapsing them would resolve a citation to the wrong part
    of a guide.
    """
    text = _clean_heading((text or "").strip().lstrip("#"))
    return " ".join(text.casefold().split())


def _match_section(declared: str, candidates: list[str]) -> str | None:
    """The candidate heading `declared` refers to, or None.

    None is the common, safe outcome: the passage had no headings, the composer left the
    field empty, or it named something that is not in this passage. In every one of those
    cases the citation survives without a section — pointing at the wrong part of a guide
    is worse than pointing at the guide.
    """
    if not declared:
        return None
    key = _normalize_section(declared)
    if not key:
        return None
    for candidate in candidates:
        if _normalize_section(candidate) == key:
            return candidate
    return None


def _application_name(meta: dict) -> str | None:
    """Display name for a chunk's application slug.

    None for a chunk with no application — the deliberate "global document" case in
    `rag_client._build_filter`. It belongs to no module, so naming one would be a guess.
    """
    slug = (meta or {}).get("application")
    if not slug:
        return None
    return APPLICATION_NAMES.get(slug, slug)


def _doc_id(meta: dict) -> str:
    """The chunk's source document id.

    The product corpus (enterprise-llm-service) writes `doc_id`; the in-repo QA indexer
    writes `source_doc_id`. Read both, exactly as `RagClient._query` does.
    """
    m = meta or {}
    return m.get("doc_id") or m.get("source_doc_id") or ""


def catalog(
    metas: list[dict] | None,
    qa_metas: list[dict] | None = None,
) -> dict[str, Citation]:
    """Every id the composer may legally declare this turn, mapped to its Citation.

    `metas` is positionally aligned with the passages handed to the composer — that is
    what `RagClient.search` returns and what `passages_block` numbers — so passage i is
    addressed as `"i"`. Q&A records are numbered separately as `"qa:i"` because they are
    numbered separately in the prompt too.

    The Citations here carry no section yet: which heading applies depends on what the
    answer used, which `select` resolves. What is fixed at this point is the fallback
    label, so a chunk that ends up with no section still has something to show.

    A chunk with no `doc_id` is left out of the catalog entirely, so declaring it fails
    the whitelist check. There is nothing to attribute it to and no way to dedupe it.
    """
    out: dict[str, Citation] = {PROCESS_DOC_ID: PROCESS_CITATION}
    for i, meta in enumerate(metas or []):
        doc_id = _doc_id(meta)
        if not doc_id:
            continue
        application = _application_name(meta)
        out[str(i)] = Citation(
            doc_id=doc_id,
            # Replaced by the section in `select` when the answer names one. Falling back
            # to the application is what keeps a heading-less chunk citable without
            # naming the file it came from.
            label=application or GENERIC_GUIDE_LABEL,
            section=None,
            application=application,
            kind="guide",
            confidence=float((meta or {}).get("confidence", 0.0) or 0.0),
        )
    for i, meta in enumerate(qa_metas or []):
        doc_id = _doc_id(meta)
        out[f"{_QA_PREFIX}{i}"] = Citation(
            # Prefixed to match how KnowledgeAgent already namespaces QA ids, and so a
            # Q&A record can never collide with a guide's UUID.
            doc_id=f"{_QA_PREFIX}{doc_id}" if doc_id else f"{_QA_PREFIX}{i}",
            label=QA_LABEL,
            section=None,
            application=_application_name(meta),
            kind="qa",
            confidence=float((meta or {}).get("confidence", 0.0) or 0.0),
        )
    return out


def _normalize_id(raw: str) -> str:
    """Tolerate the shapes a model writes for the same id.

    `[0]`, `passage 0`, and `0` all mean passage zero; `QA:1` means `qa:1`. Normalising
    here rather than tightening the prompt keeps a formatting slip from silently costing
    the user a real source — it is only ever a lookup key, and an id that still misses the
    catalog is dropped exactly as before.
    """
    text = (raw or "").strip().strip("[]() ").lower()
    text = re.sub(r"^(passage|đoạn trích|doan trich)\s*", "", text)
    return text.strip()


def _passage_text(key: str, passages: list[str], qa_passages: list[str]) -> str:
    """The text behind a normalised catalog key, or "" when there is none."""
    if key == PROCESS_DOC_ID:
        return ""
    if key.startswith(_QA_PREFIX):
        idx, source = int(key[len(_QA_PREFIX) :]), qa_passages
    else:
        idx, source = int(key), passages
    return source[idx] if 0 <= idx < len(source) else ""


def select(
    cited: list[CitedSource] | None,
    cat: dict[str, Citation],
    passages: list[str] | None = None,
    qa_passages: list[str] | None = None,
) -> list[Citation]:
    """The declared sources that this turn actually offered, resolved and deduped.

    Four things happen, and they answer the same question — what may the user be shown:
      - an id absent from `cat` is dropped. It is either invented or points at a chunk we
        cannot attribute, and a citation the user cannot verify is worse than none;
      - **only guides survive.** The process block and the CS-verified Q&A store are not
        documents the customer has: there is nothing for them to open and check, so
        naming one answers "where did this come from?" with something unresolvable, which
        is worse than staying quiet. An answer grounded solely in those shows no sources
        at all. They stay declarable — see the note below — just never displayed;
      - a declared section is kept only if that passage really contains that heading
        (`_match_section`). Otherwise the citation stays and loses its section;
      - rows that would render identically collapse. The key is `(label, application)` —
        what the user sees — not the document, because one guide contributing two
        sections is two meaningfully different rows, while two heading-less chunks from
        one application would print the same line twice.

    NOTE: the composer still DECLARES `quy_trinh_chung` and `qa:<i>`, and this function
    is the only place they are dropped. Two reasons they must stay declarable:
    `passages_for` decides from the same declarations whether the grounding judge runs at
    all, so removing the Q&A ids would skip the judge for an answer resting on a CS
    record; and taking away the process id would push
    the model to attribute a process claim to whichever passage is nearest, which is the
    fabricated citation this module exists to prevent.

    The first occurrence wins, so the order the composer used its sources in survives.
    """
    passages = passages or []
    qa_passages = qa_passages or []
    out: list[Citation] = []
    seen: set[tuple[str, str | None]] = set()
    for entry in cited or []:
        key = _normalize_id(entry.id)
        base = cat.get(key)
        if base is None or base.kind != "guide":
            continue
        section = _match_section(entry.section, sections(_passage_text(key, passages, qa_passages)))
        hit = base.model_copy(update={"section": section, "label": section or base.label})
        row = (hit.label, hit.application)
        if row in seen:
            continue
        seen.add(row)
        out.append(hit)
    return out


def passages_for(
    cited: list[CitedSource] | None,
    cat: dict[str, Citation],
    passages: list[str],
    qa_passages: list[str] | None = None,
) -> list[str]:
    """Text of the cited passages.

    KnowledgeAgent uses this as the grounding judge's GATE: a non-empty result means the
    answer stood on at least one real passage, and the judge then sees every passage the
    turn retrieved (not only these), so a wrong or missing citation cannot get a correct
    claim flagged. Empty means the judge is skipped.

    Deduped by passage, not by declaration — one chunk cited for two of its sections is
    still one passage, and handing the judge the same text twice would waste tokens.

    `quy_trinh_chung` contributes nothing here — it is not a passage. The judge receives
    the process block in its own system prefix instead, the same way the composer did.
    """
    qa_passages = qa_passages or []
    out: list[str] = []
    seen: set[str] = set()
    for entry in cited or []:
        key = _normalize_id(entry.id)
        if key == PROCESS_DOC_ID or key not in cat or key in seen:
            continue
        seen.add(key)
        text = _passage_text(key, passages, qa_passages)
        if text:
            out.append(text)
    return out

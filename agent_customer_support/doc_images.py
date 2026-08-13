"""Document images in answers: passage refs → scoped markers → presigned URLs.

The user guides were converted from .docx with pandoc, so their chunks carry inline
image references — `![](media/image23.png)` — pointing at screenshots and button glyphs
from the original document. This module is the whole text transform that turns those into
images the user actually sees. It is pure: no I/O, so every rule below is testable without
S3 or an LLM.

Three invariants decide what a reply shows, and nothing else does. No application, slug,
or document is named anywhere in this file:

1. **The key is derived, never mapped.** `<prefix>/<application>/<name>`, where the
   application slug comes off the retrieved chunk's metadata. `media/imageNN.png` is
   unique only *within* one document — every guide has its own `image1.png` — so the slug
   is not decoration, it is what makes the name resolvable at all.

2. **A ref survives only if the object exists.** `rewrite_passages` takes an availability
   set per slug and drops anything absent. A cited document with no images in the bucket
   therefore yields no markers and a plain-text answer, through the same code path as one
   that has them. There is no "has images" flag to maintain.

3. **A ref never survives unresolved.** Every `media/…` ref is either rewritten to a
   resolvable marker or deleted. Leaving one in place would let the composer copy it
   through verbatim and the widget would render a broken relative URL.

Composition adds a fourth rule: only markers this module handed to the composer can come
back out (`select`). A model that invents an image number produces a marker that is not
in the catalog, and it is stripped before the reply is ever persisted.
"""

import logging
import re
from typing import Callable, Iterable, Literal, Mapping

from agent_customer_support.config import get_settings

logger = logging.getLogger(__name__)

Kind = Literal["screen", "icon"]

# The pandoc-emitted reference as it appears in chunk text. Alt text is almost always
# empty in these documents but is tolerated. Only .png occurs across all 134 refs.
_REF_RE = re.compile(r"!\[[^\]]*\]\(media/(image\d+\.png)\)")

# The scoped marker the composer sees, following the [[…]] convention already used for
# routing markers in agents/knowledge.py. Kind is baked in so every later stage —
# presigning, re-rendering a stored turn — needs the marker and nothing else.
_MARKER_RE = re.compile(r"\[\[img:(screen|icon):([a-z0-9_]+)/(image\d+\.png)\]\]")

# Any [[img:…]]-shaped token, including malformed or invented ones. `select` sweeps this
# and validates with _MARKER_RE, so a near-miss (wrong charset, missing kind, made-up
# path) is deleted rather than shown to the user as literal text.
_MARKER_SHAPED_RE = re.compile(r"\[\[img:[^\]]*\]\]")


def s3_key(slug: str, name: str) -> str:
    return f"{get_settings().doc_images_prefix}/{slug}/{name}"


def marker(kind: Kind, slug: str, name: str) -> str:
    return f"[[img:{kind}:{slug}/{name}]]"


def _kind_for(line: str, ref: str) -> Kind:
    """Classify a ref by its position in the source markdown line.

    In these guides the distinction is structural, not a guess: a screenshot sits alone
    on its own line, while a button glyph sits inside a table row describing it
    (`| ![](media/image5.png) | Chỉnh sửa: Mở form… |`). Deriving kind here costs nothing,
    where an object-size check would cost an S3 HEAD per image on the request path.
    """
    return "screen" if line.strip() == ref else "icon"


def _rewrite_line(line: str, slug: str, names: set[str]) -> str:
    def replace(m: re.Match) -> str:
        name = m.group(1)
        if name not in names:
            return ""
        return marker(_kind_for(line, m.group(0)), slug, name)

    return _REF_RE.sub(replace, line)


def slugs_with_refs(passages: list[str], metas: list[dict]) -> set[str]:
    """Application slugs whose retrieved passages actually contain image refs.

    Lets the caller look up availability only for documents that could use it, so the 10
    guides carrying no images never trigger an S3 listing.
    """
    slugs: set[str] = set()
    for i, passage in enumerate(passages):
        if not _REF_RE.search(passage):
            continue
        meta = metas[i] if i < len(metas) else {}
        slug = (meta or {}).get("application") or ""
        if slug:
            slugs.add(slug)
    return slugs


def rewrite_passages(
    passages: list[str],
    metas: list[dict],
    available: Mapping[str, Iterable[str]],
) -> list[str]:
    """Rewrite `media/…` refs into scoped markers, dropping every ref that can't resolve.

    `metas` is positionally aligned with `passages` — that is exactly what
    `RagClient.search` returns — so passage i is scoped by metas[i]["application"].
    `available` maps a slug to the names present in the bucket (see `slugs_with_refs`).

    A ref is dropped rather than rewritten when the chunk carries no application (the
    deliberate "global document" case in rag_client._build_filter: no application means
    visible to everyone, but it also means no prefix to resolve against) or when the file
    is not in the bucket. Either way the passage still reaches the composer, just without
    a picture it could have offered.

    Passages with no refs at all are returned untouched — the 10 guides that carry no
    images must come out byte-identical.
    """
    out: list[str] = []
    for i, passage in enumerate(passages):
        if not _REF_RE.search(passage):
            out.append(passage)
            continue
        meta = metas[i] if i < len(metas) else {}
        slug = (meta or {}).get("application") or ""
        names = set(available.get(slug) or ()) if slug else set()
        out.append("\n".join(_rewrite_line(ln, slug, names) for ln in passage.split("\n")))
    return out


def _is_screen(mk: str) -> bool:
    m = _MARKER_RE.match(mk)
    return m is not None and m.group(1) == "screen"


def select(
    text: str,
    available: Mapping[str, Iterable[str]],
    max_images: int = 5,
) -> str:
    """Keep only offered, deduped markers, capped — and delete every other candidate.

    Four things happen in one pass, because they all answer the same question, "which
    markers may reach the user":
      - a marker that is malformed is deleted;
      - a marker naming a file that is not in `available` is deleted. **Shape is not
        enough**: a model that invents `image999.png` under a real slug writes a perfectly
        well-formed marker, and signing a URL for it would render a broken image. The
        catalog is the whitelist, so `available` must be the same mapping the passages
        were rewritten against;
      - a marker repeated across the reply is kept at its first position only;
      - past `max_images`, screenshots win over icons, since a screen carries far more of
        the answer than a button glyph does.

    Position is preserved for survivors: the composer placed each marker next to the step
    it illustrates, which is the entire point. Text with no marker-shaped token is
    returned unchanged, so a reply from an image-less document is untouched.
    """
    if not _MARKER_SHAPED_RE.search(text):
        return text

    # Dedupe by marker, first occurrence wins. Unknown files never enter the pool.
    unique: dict[str, int] = {}
    for m in _MARKER_RE.finditer(text):
        if m.group(3) not in set(available.get(m.group(2)) or ()):
            continue
        unique.setdefault(m.group(0), m.start())

    if max_images > 0 and unique:
        ranked = sorted(
            unique.items(),
            # screen before icon, then document order. Sorting is stable, so the position
            # key only breaks ties within a kind.
            key=lambda kv: (0 if _is_screen(kv[0]) else 1, kv[1]),
        )
        keep = {mk for mk, _ in ranked[:max_images]}
    else:
        keep = set()

    seen: set[str] = set()

    def prune(m: re.Match) -> str:
        mk = m.group(0)
        if mk in keep and mk not in seen:
            seen.add(mk)
            return mk
        return ""

    return _tidy(text, _MARKER_SHAPED_RE.sub(prune, text))


def markers_in(text: str) -> list[tuple[str, str, str]]:
    """Every valid `(kind, slug, name)` in `text`, deduped, in order.

    Exists so an async caller can resolve all its URLs first and then run the
    substitution pass, which is synchronous.
    """
    out: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for m in _MARKER_RE.finditer(text):
        key = (m.group(2), m.group(3))
        if key in seen:
            continue
        seen.add(key)
        out.append((m.group(1), m.group(2), m.group(3)))
    return out


def presign_markers(text: str, presign: Callable[[str, str], str]) -> str:
    """Swap each marker for markdown `![kind](url)`.

    Applied to the outbound reply only — the persisted turn keeps markers. Presigned URLs
    expire, so storing one would archive a dead link and (worse) feed 500 characters of
    signature into the transcript the LLM re-reads next turn. This mirrors the existing
    StoredAttachment/AttachmentRef split for user uploads.

    `kind` rides in the alt text because that is the only channel surviving into rendered
    markdown, and the widget needs it to choose inline glyph vs clickable preview. The
    widget supplies its own human-readable alt label.

    A marker whose URL cannot be signed is dropped: a missing picture beats a broken one.
    """
    if not _MARKER_RE.search(text):
        return text

    def replace(m: re.Match) -> str:
        kind, slug, name = m.group(1), m.group(2), m.group(3)
        try:
            url = presign(slug, name)
        except Exception as exc:  # noqa: BLE001 - degrade, never break a generated reply
            logger.warning("presign failed for %s/%s, dropping image: %s", slug, name, exc)
            return ""
        return f"![{kind}]({url})"

    return _tidy(text, _MARKER_RE.sub(replace, text))


def strip(text: str) -> str:
    """Remove every marker. The fallback when images cannot be resolved at all."""
    if not _MARKER_SHAPED_RE.search(text):
        return text
    return _tidy(text, _MARKER_SHAPED_RE.sub("", text))


def _tidy(before: str, after: str) -> str:
    """Clean up the whitespace a removed marker leaves behind.

    Deleting a marker can leave a double space mid-sentence, or an entirely empty line
    where a standalone screenshot was. Both are visible in the chat bubble.

    Comparing against `before` is what keeps this from reformatting the composer's own
    prose: a line is dropped only if it is blank *now* and was not blank before. The
    author's deliberate paragraph breaks survive; the holes left by removal do not.
    Substitution never changes the line count, so the two zip.
    """
    out: list[str] = []
    for old, new in zip(before.split("\n"), after.split("\n")):
        new = re.sub(r"[ \t]{2,}", " ", new).rstrip()
        if not new and old.strip():
            continue  # this line held only markers
        out.append(new)
    return "\n".join(out).strip()

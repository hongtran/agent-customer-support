# Q&A Learning Loop — Spec 2b: Second-class Retrieval Merge

**Date:** 2026-06-22
**Status:** Approved design — ready for implementation plan
**Spec:** 2b of 2 (consumes the `qa` collection produced by Spec 2a).
**Builds on:** Spec 1 (in-repo Qdrant read path), Spec 2a (Q&A capture/curation/indexing).

## Goal

Make the agent answer from the curated Q&A corpus. `KnowledgeAgent` retrieves
from the `qa` collection alongside the product guide and composes a single
answer with confidence-gated precedence: a strongly-matching, CS-verified Q&A
leads (so it can fix wrong guide answers); otherwise the guide leads and Q&A is
supplementary.

## Non-goals

- Any change to capture/curation/indexing (Spec 2a) or the read path (Spec 1).
- A new retrieval implementation — the `qa` collection shares the product
  payload layout, so retrieval reuses `RagClient.search` unchanged.
- Embedding the query once for both searches (a future optimization; 2b accepts
  two embeds).
- Showing an explicit "CS-verified" label to the end user (the reply stays
  seamless).

## Precedence model (decided)

Confidence-gated, three-tier:

1. **CS-verified answer, when it leads** — the `qa` search's `top_confidence ≥
   qa_lead_threshold` (default `0.85`). Highest precedence: it overrides the
   guide on conflict (this is what makes the 👎-feedback correction loop work).
2. **QUY TRÌNH (process)** — cross-module sequence/conditions/permissions.
3. **ĐOẠN TRÍCH (guide passages)** — in-module detail.

When `qa` passages exist but do **not** clear the threshold, they are
**supplementary** (below the guide), not authoritative.

## Data flow

```
query
  → search(product collection)          (existing)
  → search(qa collection)               (new; same applications filter)
  → qa_leads = qa.top_confidence >= qa_lead_threshold and qa has passages
  → _compose(guide_passages, qa_passages, qa_leads)   → reply
```

All other `KnowledgeAgent` behavior is unchanged: the clarify loop,
`[[suspected_bug]]`, and the miss→capture path. If `qa` now answers, `_compose`
simply won't emit `[[no_answer]]`, so no miss/capture fires — automatically.

## Components

### 1. Retrieval merge (`agents/knowledge.py`, at the current single-search site)

- Add a second search:
  `qa_res = await _safe_qa_search(ctx, query, applications)` where
  `_safe_qa_search` calls
  `ctx.rag.search(query, collection=cfg.qa_collection, applications=applications)`
  and returns an **empty result** (`{"passages": [], "citations": [],
  "top_confidence": 0.0}`) on **any** exception.
  - **Rationale:** the `qa` collection does not exist until the first approval
    (Spec 2a's `QAIndexer.ensure_collection` creates it on first upsert), so a
    missing collection (or any Qdrant error) must degrade to product-only. The
    guide path must never break because of Q&A.
- `qa_passages = qa_res["passages"]`; `qa_citations = qa_res["citations"]`.
- `qa_leads = bool(qa_passages) and qa_res["top_confidence"] >= cfg.qa_lead_threshold`.
- Citations: merge product citations with qa citations, the latter prefixed
  `qa:` (e.g. `qa:<record_id>`) so the source is distinguishable in
  `ChatResponse.citations`.
- Pass `qa_passages` and `qa_leads` into `_compose`.

### 2. Composition (`agents/knowledge.py::_compose` + `agents/prompts.py`)

`_compose` gains `qa_passages: list[str]` and `qa_leads: bool`.

- **No qa passages →** behavior is **byte-identical to today**: the existing
  `KNOWLEDGE_COMPOSE_PROMPT` (two-source) is used, content is
  `history + question + ĐoạnTrích(guide)`. Zero regression for the common case.
- **qa passages present →** use the three-source variant
  `KNOWLEDGE_COMPOSE_PROMPT_WITH_QA`, and append a CS-answer block to the
  user-content, labeled and marked authoritative-or-supplementary based on
  `qa_leads`:
  - `qa_leads = True`: the block is headed e.g.
    `ĐÁP ÁN CS XÁC NHẬN (ưu tiên cao nhất cho câu hỏi này):` and lists the qa
    passages.
  - `qa_leads = False`: headed e.g.
    `ĐÁP ÁN CS THAM KHẢO (bổ trợ):`.
- The `allow_clarify=False` resume suffix (`KNOWLEDGE_RESUME_NO_CLARIFY`) and
  the `PROCESS_BLOCK` cached system prefix are unchanged and apply to both
  variants.

### 3. Prompt: `KNOWLEDGE_COMPOSE_PROMPT_WITH_QA` (`agents/prompts.py`, new)

A copy of `KNOWLEDGE_COMPOSE_PROMPT` with exactly these deltas (so the tuned
behavior — diagnosis, clarify, admin-routing, anti-hallucination — is otherwise
preserved verbatim):

- "Chỉ dựa trên **hai nguồn**" → "**ba nguồn**".
- Add a third entry to the `NGUỒN` list:
  **3. ĐÁP ÁN CS XÁC NHẬN** — câu trả lời do nhân viên CS biên soạn/duyệt cho
  đúng câu hỏi này (đã được người thật kiểm chứng). Có thể được đánh dấu "ưu
  tiên cao nhất" (đáp án đúng cho câu hỏi) hoặc "bổ trợ".
- Replace the 2-way precedence line (L104) with a **3-tier hierarchy**, adding
  the CS tier on top without contradicting the existing guide/passage rule:
  *Khi các nguồn mâu thuẫn: (1) ĐÁP ÁN CS XÁC NHẬN được đánh dấu "ưu tiên cao
  nhất" thắng tất cả (kể cả QUY TRÌNH) cho đúng câu hỏi đó; (2) QUY TRÌNH chuẩn
  cho trình tự liên-module/điều kiện/phân quyền/điểm kiểm soát; (3) ĐOẠN TRÍCH
  chuẩn cho chi tiết nội bộ module. ĐÁP ÁN CS "bổ trợ" chỉ tham khảo, không vượt
  QUY TRÌNH.*
- Anti-hallucination line: "Không dùng kiến thức ngoài **ba nguồn**."
- `[[no_answer]]` marker: "**Tất cả các nguồn** đều không trả lời được".

The anti-hallucination contract is preserved: Q&A is a *named, human-vetted*
source, not "outside knowledge."

### 4. Config (`config.py`)

`qa_lead_threshold: float = 0.85`.

## Error handling

- `qa` search failure (missing collection / Qdrant down) → silent degrade to
  product-only via `_safe_qa_search` (logged, not surfaced). The guide answer is
  never blocked by Q&A availability.
- No new user-facing error paths.

## Testing

- **qa leads:** `ctx.rag.search` mocked to return high-`top_confidence` qa
  results for the qa collection; assert `_compose` is invoked with `qa_leads=True`
  and that the user-content includes the authoritative CS block and uses
  `KNOWLEDGE_COMPOSE_PROMPT_WITH_QA` (assert via mocked `complete_text` system/
  content args).
- **qa supplementary:** qa results below threshold → `_compose` called with
  `qa_leads=False`, block headed as supplementary, three-source prompt used.
- **no qa:** qa search returns empty → the **two-source** prompt is used and the
  content is byte-identical to the pre-2b path (guard the no-regression case).
- **qa collection missing → graceful:** `ctx.rag.search` raises for the qa
  collection → `_safe_qa_search` returns empty, the turn still answers from the
  guide, no exception propagates.
- **applications filter:** the qa search is called with the same `applications`
  value as the product search.
- **citations:** merged result contains product citations plus `qa:`-prefixed qa
  citations.
- Mock `ctx.rag.search` to dispatch on the `collection` kwarg (product vs qa).

## Rollout

Behavior is inert until the `qa` collection has approved content: with an empty/
absent collection every turn degrades to product-only and uses the unchanged
two-source prompt. As CS approves Q&A, those questions start being answered
(supplementary first, authoritative once they clear the threshold) with no
further deploys.

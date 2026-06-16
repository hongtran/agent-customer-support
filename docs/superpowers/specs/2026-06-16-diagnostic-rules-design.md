# Diagnostic Operating Rules in the Knowledge Agent

**Date:** 2026-06-16
**Status:** Design — pending review
**Branch:** feat/multi-agent-rearchitecture

## Problem

The CenLab operations guide (`Huong_dan_van_hanh_Cenlab.pdf`) contains cross-cutting
*operating principles* — e.g. "thiếu dữ liệu → kiểm tra master data trước",
"không có quyền thao tác → liên hệ admin để được phân quyền". These are already indexed
into the `cenlab` RAG collection, but indexing alone does not make the agent apply them
reliably.

The reason is structural. `KNOWLEDGE_COMPOSE_PROMPT` is deliberately grounded with a hard
anti-hallucination rule (`prompts.py:49`): the answer may only use retrieved passages. So a
principle surfaces *only if the retriever pulls that exact chunk*. For the two symptoms above
this fails in the worst way — retrieval **succeeds wrongly**:

- *"tôi không thấy dữ liệu khách hàng"* → retriever returns the customer-module how-to
  (`present=True`), compose answers with click-steps into a screen that is empty because master
  data was never set up. Root cause never surfaced.
- *"không thao tác được menu X"* → retriever returns the how-to for X, telling the user to click
  a menu their role can't see.

Because `present=True` in both cases, a "fallback after a RAG miss" would never fire. The rule
layer must be able to **lead or override even when RAG thinks it has an answer.**

## Scope

**In scope (confirmed):** a focused, *reactive* set of symptom→guidance diagnostic rules.
Reactive = a rule fires only when the user's symptom matches it; no match leaves the existing
pipeline byte-for-byte unchanged.

**Out of scope (this pass):** Part 2 pre-op principles as always-on context; Part 5 special-case
workflows; the 8-stage process ordering / control points (B1–B21) — that overlaps the Flow engine.

## Seed rule set

Core (ship these):

| id | symptom | guidance (canonical VN, source) |
|---|---|---|
| `missing_master_data` | danh mục/dropdown trống, "không thấy dữ liệu", không tìm thấy mục để chọn | Kiểm tra & chuẩn hoá master data (danh mục nền tảng) trước khi phát sinh nghiệp vụ — đây là dữ liệu dùng chung cho toàn luồng. (Part 2.2) |
| `no_permission` | "không có quyền", menu bị ẩn, không thực hiện được thao tác | Liên hệ quản trị hệ thống/admin để được rà soát & phân quyền phù hợp với mục đích và vị trí công việc. (Part 2.1) |
| `ui_not_configured` | không thấy cột/trường/thông tin cần xem trên màn hình | Thiết lập lại giao diện theo người dùng: cấu hình hiển thị đúng thông tin mình quan tâm. (Part 2.3) |

Candidates (confirm during implementation, easy to add as data — same shape):
`forgot_lay_mau_button` (B7-A: PQT không nhận được luồng vì chưa nhấn LẤY MẪU),
`nghiem_thu_blocked` (B19: chưa tạo được nghiệm thu vì PYC chưa xuất đủ kết quả),
`edit_after_handoff` (Part 5.5: sửa dữ liệu sau khi đã chuyển giao bước).

The seed set is intentionally small. The whole point of rules-as-data is that growing the set is
a data edit, not a code change.

## Architecture (Approach C)

Three units, each independently testable.

### 1. `agents/diagnostics.py` — rules as data
```python
@dataclass(frozen=True)
class DiagnosticRule:
    id: str
    symptom: str      # natural-language description, fed to the classifier
    guidance: str     # canonical Vietnamese answer text

DIAGNOSTIC_RULES: list[DiagnosticRule] = [...]
RULES_BY_ID: dict[str, DiagnosticRule] = {r.id: r for r in DIAGNOSTIC_RULES}
```
Pure data + lookup. No I/O. Trivially testable / reviewable.

### 2. Classifier — one cheap LLM call
New `DIAGNOSTIC_PROMPT` in `prompts.py`. A method on `KnowledgeAgent`:
```python
async def _diagnose(self, query: str, cfg: Settings) -> DiagnosticRule | None
```
- Input: the **contextualized** query (pronouns/image already resolved).
- One `complete_text` call, `system=DIAGNOSTIC_PROMPT`, `model=cfg.model_for("diagnostic")`.
- Prompt lists each rule `id` + `symptom`; model returns JSON `{"rule_id": "<id>" | "none"}`.
- **Fail-closed:** any parse error or unknown id → `None` (behave as if no rule matched).
- Returns the `DiagnosticRule` or `None`.

Per the chosen cost option (a): the classifier runs on **every** knowledge turn (no pre-filter,
no folding into another call) — simplest and most reliable. Cost is bounded by routing it to a
cheap model via a new `diagnostic_model` setting.

### 3. Injection into compose
In `KnowledgeAgent.run`, after contextualize, run `_diagnose` **in parallel** with `rag.search`
(both are async; gather them) so the rule can lead even when RAG returns a (wrong) answer.

If a rule matched:
- Build a synthetic passage: `f"[OP] {rule.guidance}"` and **prepend** it to `passages`.
- Force `present = True` (so RAG-miss symptom cases still answer instead of clarify/handoff).
- Compose as normal; the `[OP]` passage is now part of the grounded context.

Amend `KNOWLEDGE_COMPOSE_PROMPT` with one line:
> *Nếu trong đoạn trích có mục bắt đầu bằng `[OP]` (nguyên tắc vận hành), hãy nêu nguyên tắc/
> hướng kiểm tra đó TRƯỚC, rồi mới bổ sung hướng dẫn chi tiết từ các đoạn còn lại (nếu có).*

This blends rule + how-to naturally: *"Trước hết hãy liên hệ admin để được phân quyền; sau khi có
quyền, thao tác như sau…"* The `[OP]` text is itself the grounding source, so the anti-hallucination
contract is preserved — we are not inventing knowledge, we are injecting an authoritative passage.

### 4. Config
Add to `Settings` (`config.py`):
```python
diagnostic_model: str | None = "gpt-4o-mini"
```
`model_for("diagnostic")` resolves it via the existing `getattr` mechanism — no change to
`model_for` needed. Default `gpt-4o-mini` matches the existing cheap-model siblings; the user may
point it at `claude-haiku-4-5-20251001` via env. (Recommended cheap model: Haiku.)

## Data flow

```
ctx.message
  → _contextualize (existing)
  → gather( rag.search , _diagnose )        ← NEW parallel branch
  → rule? prepend "[OP] guidance"; present=True
  → _present (existing, short-circuited True when a rule matched)
  → _compose (existing prompt + one [OP] line)
  → parse_markers → AgentResult
```
No-match path is identical to today.

## Error handling

- Classifier unparseable / unknown id / LLM error → `None`. The turn proceeds exactly as it does
  today. A diagnostic failure can never *block* an answer the pipeline would otherwise give.
- Injection never removes RAG passages; it only prepends. If RAG also found a relevant how-to, the
  user gets both, in the right order.

## Testing

- **Rules data:** every rule has unique id, non-empty symptom + guidance; `RULES_BY_ID` round-trips.
- **Classifier (`_diagnose`):** mock `complete_text`. For each seed rule, ≥2 phrasings classify to
  the right id. Unrelated query → `none`. Malformed JSON → `None` (fail-closed). Unknown id → `None`.
- **Injection (`run`):** with a forced match (mock `_diagnose`), assert `[OP]` passage is first,
  `present` is True even when `rag.search` returns no passages, and the matched guidance reaches
  `_compose`. With no match, assert the call sequence/result is unchanged vs. baseline.
- **Compose prompt:** a focused test that, given an `[OP]` passage + a module passage, the composed
  answer leads with the principle (can be asserted loosely / via mock).

## Risks & tradeoffs

- **+1 LLM call per knowledge turn.** Accepted (option a) for reliability; bounded by cheap model.
  If cost becomes a concern, the keyword pre-filter (approach B) can gate it later without redesign.
- **False positives** (a rule fires when it shouldn't): mitigated by the classifier seeing the
  fully-contextualized query and by injection being *additive* — RAG passages remain, so a wrong
  `[OP]` degrades to a slightly-off preamble, not a wrong answer.
- **Rule/RAG conflict:** if a future rule contradicts a doc, the `[OP]`-first instruction makes the
  rule win. That is intended for operating principles, but worth remembering when adding rules.

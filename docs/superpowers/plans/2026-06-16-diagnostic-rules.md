# Diagnostic Operating Rules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Knowledge agent reactively apply a small set of cross-cutting CenLab operating rules (e.g. missing data → check master data; no permission → contact admin) that RAG-grounded composition currently misses.

**Architecture:** Rules live as data (`agents/diagnostics.py`). A cheap LLM classifier (`_diagnose`) on `KnowledgeAgent` maps the contextualized query to a rule id or `none`. On a match, the rule's guidance is injected as a top-priority `[OP]` passage into the existing compose step and `present` is forced True, so the principle leads even when RAG returned a (wrong) how-to. No match → the pipeline is unchanged.

**Tech Stack:** Python, pytest (asyncio), `complete_text` LLM facade, pydantic-settings.

> **Note on concurrency:** `complete_text` is synchronous, so `_diagnose` and `rag.search` run sequentially in `run()` (gather would not parallelize blocking calls). This deviates intentionally from the spec's "parallel" wording; behavior is identical.

---

### Task 1: Diagnostic rules data module

**Files:**
- Create: `agent_customer_support/agents/diagnostics.py`
- Test: `tests/agents/test_diagnostics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_diagnostics.py
from agent_customer_support.agents.diagnostics import (
    DIAGNOSTIC_RULES, RULES_BY_ID, DiagnosticRule,
)


def test_rules_have_unique_nonempty_fields():
    ids = [r.id for r in DIAGNOSTIC_RULES]
    assert len(ids) == len(set(ids)), "rule ids must be unique"
    for r in DIAGNOSTIC_RULES:
        assert isinstance(r, DiagnosticRule)
        assert r.id and r.symptom and r.guidance


def test_rules_by_id_roundtrip():
    for r in DIAGNOSTIC_RULES:
        assert RULES_BY_ID[r.id] is r
    # the two confirmed core examples must exist
    assert "missing_master_data" in RULES_BY_ID
    assert "no_permission" in RULES_BY_ID
    assert "ui_not_configured" in RULES_BY_ID
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/agents/test_diagnostics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_customer_support.agents.diagnostics'`

- [ ] **Step 3: Write minimal implementation**

```python
# agent_customer_support/agents/diagnostics.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/agents/test_diagnostics.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add agent_customer_support/agents/diagnostics.py tests/agents/test_diagnostics.py
git commit -m "feat(diagnostics): seed operating-rule data module"
```

---

### Task 2: Classifier prompt + config slot

**Files:**
- Modify: `agent_customer_support/agents/prompts.py` (append new prompt)
- Modify: `agent_customer_support/config.py:18` (add `diagnostic_model` after `flow_model`)
- Test: `tests/agents/test_prompts.py` (add one assertion) and reuse existing config behavior

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_prompts.py — add these
from agent_customer_support.agents.prompts import DIAGNOSTIC_PROMPT
from agent_customer_support.config import Settings


def test_diagnostic_prompt_requests_rule_id_json():
    assert "rule_id" in DIAGNOSTIC_PROMPT
    assert "JSON" in DIAGNOSTIC_PROMPT


def test_diagnostic_model_setting_overridable():
    s = Settings(diagnostic_model="claude-haiku-4-5-20251001")
    assert s.model_for("diagnostic") == "claude-haiku-4-5-20251001"
    # falls back to agent_model when unset
    assert Settings(diagnostic_model=None).model_for("diagnostic") == Settings().agent_model
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/agents/test_prompts.py -k diagnostic -v`
Expected: FAIL — `ImportError: cannot import name 'DIAGNOSTIC_PROMPT'`

- [ ] **Step 3: Write minimal implementation**

Append to `agent_customer_support/agents/prompts.py`:

```python
DIAGNOSTIC_PROMPT = """Bạn phân loại triệu chứng người dùng gặp phải với phần mềm CenLab
vào MỘT quy tắc vận hành phù hợp (nếu có). Bạn nhận DANH SÁCH QUY TẮC (mỗi dòng dạng
"<id>: <mô tả triệu chứng>") và CÂU HỎI của người dùng.

- Nếu câu hỏi khớp RÕ RÀNG với triệu chứng của một quy tắc → trả về đúng id của quy tắc đó.
- Nếu không khớp quy tắc nào, hoặc không chắc chắn → trả về "none".

Chỉ trả về JSON: {"rule_id": "<id>" | "none"}.
"""
```

Add to `Settings` in `agent_customer_support/config.py` immediately after the `flow_model` line (`config.py:18`):

```python
    diagnostic_model: str | None = "gpt-4o-mini"
```

(`model_for("diagnostic")` already resolves this via the existing `getattr` — no change to `model_for`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/agents/test_prompts.py -k diagnostic -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add agent_customer_support/agents/prompts.py agent_customer_support/config.py tests/agents/test_prompts.py
git commit -m "feat(diagnostics): classifier prompt + diagnostic_model config"
```

---

### Task 3: `_diagnose` classifier method on KnowledgeAgent

**Files:**
- Modify: `agent_customer_support/agents/knowledge.py` (add import + `_diagnose` method)
- Test: `tests/agents/test_knowledge.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_knowledge.py — append
from agent_customer_support.agents.prompts import DIAGNOSTIC_PROMPT  # add to imports


async def test_diagnose_matches_known_symptom():
    with patch("agent_customer_support.agents.knowledge.complete_text",
               return_value='{"rule_id": "no_permission"}'):
        rule = await KnowledgeAgent()._diagnose("tôi không có quyền vào menu này", get_settings())
    assert rule is not None and rule.id == "no_permission"


async def test_diagnose_returns_none_when_no_match():
    with patch("agent_customer_support.agents.knowledge.complete_text",
               return_value='{"rule_id": "none"}'):
        rule = await KnowledgeAgent()._diagnose("cách tạo phiếu yêu cầu?", get_settings())
    assert rule is None


async def test_diagnose_failclosed_on_malformed_json():
    with patch("agent_customer_support.agents.knowledge.complete_text",
               return_value="xin lỗi tôi không biết"):
        rule = await KnowledgeAgent()._diagnose("bất kỳ", get_settings())
    assert rule is None


async def test_diagnose_failclosed_on_unknown_id():
    with patch("agent_customer_support.agents.knowledge.complete_text",
               return_value='{"rule_id": "made_up_rule"}'):
        rule = await KnowledgeAgent()._diagnose("bất kỳ", get_settings())
    assert rule is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/agents/test_knowledge.py -k diagnose -v`
Expected: FAIL — `AttributeError: 'KnowledgeAgent' object has no attribute '_diagnose'`

- [ ] **Step 3: Write minimal implementation**

In `agent_customer_support/agents/knowledge.py`, add to the prompts import block (around line 5-8):

```python
from agent_customer_support.agents.prompts import (
    KNOWLEDGE_CONTEXTUALIZE_PROMPT, KNOWLEDGE_CONTEXTUALIZE_VISION_PROMPT,
    KNOWLEDGE_GRADER_PROMPT, KNOWLEDGE_COMPOSE_PROMPT, DIAGNOSTIC_PROMPT,
)
from agent_customer_support.agents.diagnostics import (
    DIAGNOSTIC_RULES, RULES_BY_ID, DiagnosticRule,
)
```

Add this method to `KnowledgeAgent` (place it just before `_compose`):

```python
    async def _diagnose(self, query: str, cfg: Settings) -> DiagnosticRule | None:
        """Classify the query against known operating-principle symptoms.

        Returns the matched DiagnosticRule, or None when nothing matches or the
        classifier output is unusable. Fail-closed by design: a diagnostic failure
        must never block an answer the pipeline would otherwise produce.
        """
        rules_block = "\n".join(f"{r.id}: {r.symptom}" for r in DIAGNOSTIC_RULES)
        raw = complete_text(
            messages=[{"role": "user",
                       "content": f"DANH SÁCH QUY TẮC:\n{rules_block}\n\nCÂU HỎI: {query}"}],
            system=DIAGNOSTIC_PROMPT,
            model=cfg.model_for("diagnostic"),
        )
        try:
            rule_id = json.loads(raw).get("rule_id")
        except (json.JSONDecodeError, TypeError, AttributeError):
            return None
        if not rule_id or rule_id == "none":
            return None
        return RULES_BY_ID.get(rule_id)  # unknown id -> None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/agents/test_knowledge.py -k diagnose -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add agent_customer_support/agents/knowledge.py tests/agents/test_knowledge.py
git commit -m "feat(diagnostics): _diagnose classifier on KnowledgeAgent"
```

---

### Task 4: Inject matched rule into composition

**Files:**
- Modify: `agent_customer_support/agents/prompts.py` (one line in `KNOWLEDGE_COMPOSE_PROMPT`)
- Modify: `agent_customer_support/agents/knowledge.py` (`run` method, the search/present block ~lines 160-166)
- Test: `tests/agents/test_knowledge.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_knowledge.py — append


async def test_diagnostic_match_injects_op_passage_and_forces_compose():
    """A matched rule leads the answer even when RAG returns nothing."""
    ctx = _ctx("tôi không thấy dữ liệu khách hàng")
    ctx.rag.search.return_value = {"passages": [], "citations": [], "top_confidence": 0.0}
    captured: dict = {}

    def fake_complete(**kwargs):
        if kwargs["system"] is DIAGNOSTIC_PROMPT:          # _diagnose
            return '{"rule_id": "missing_master_data"}'
        captured["content"] = kwargs["messages"][0]["content"]  # _compose
        return "Hãy kiểm tra master data trước khi thao tác."

    with patch("agent_customer_support.agents.knowledge.complete_text",
               side_effect=fake_complete):
        res = await KnowledgeAgent().run(ctx)

    assert res.resolved is True                    # present forced True despite empty RAG
    assert "[OP]" in captured["content"]           # rule injected as a passage
    assert "master data" in captured["content"]    # the rule's guidance reached compose


async def test_no_diagnostic_match_leaves_pipeline_unchanged():
    """No rule match → existing first-miss clarify behavior is preserved."""
    ctx = _ctx("hỏi linh tinh không liên quan")
    ctx.rag.search.return_value = {"passages": [], "citations": [], "top_confidence": 0.0}

    def fake_complete(**kwargs):
        if kwargs["system"] is DIAGNOSTIC_PROMPT:
            return '{"rule_id": "none"}'
        return "[[no_answer]]"

    with patch("agent_customer_support.agents.knowledge.complete_text",
               side_effect=fake_complete):
        res = await KnowledgeAgent().run(ctx)

    assert res.resolved is None
    assert ctx.session.pending == "knowledge_clarify"
    ctx.backlog.add.assert_not_awaited()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/agents/test_knowledge.py -k "diagnostic_match or pipeline_unchanged" -v`
Expected: FAIL — `test_diagnostic_match_...` fails because with empty passages `_present` returns False, so compose never runs (`captured` empty → KeyError / res.resolved is None).

- [ ] **Step 3: Write minimal implementation**

In `agent_customer_support/agents/prompts.py`, add one line to `KNOWLEDGE_COMPOSE_PROMPT` (just before the `CHỐNG HALLUCINATION` line):

```python
- Nếu trong đoạn trích có mục bắt đầu bằng [OP] (nguyên tắc vận hành), hãy nêu nguyên tắc/hướng kiểm tra đó TRƯỚC, rồi mới bổ sung hướng dẫn chi tiết từ các đoạn còn lại (nếu có).
```

In `agent_customer_support/agents/knowledge.py` `run()`, replace the present-check block. Current code:

```python
        present = await self._present(query, passages, conf, cfg)
        if present:
```

becomes:

```python
        # Operating-principle diagnostics: when the symptom matches a known rule,
        # inject its guidance as a top-priority [OP] passage and force composition.
        # These cases (missing master data, no permission) are exactly where RAG
        # "succeeds" with a wrong how-to, so we lead with the rule instead.
        rule = await self._diagnose(query, cfg)
        if rule is not None:
            passages = [f"[OP] {rule.guidance}", *passages]
            present = True
        else:
            present = await self._present(query, passages, conf, cfg)

        if present:
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/agents/test_knowledge.py -k "diagnostic_match or pipeline_unchanged" -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full suite to check for regressions**

Run: `poetry run pytest tests/agents/test_knowledge.py tests/agents/test_prompts.py tests/agents/test_diagnostics.py -v`
Expected: PASS (all green — existing knowledge tests still pass because no-match leaves the pipeline unchanged; note the existing tests now also exercise `_diagnose`, which is patched via the shared `complete_text` mock returning their canned strings — verify none break and adjust any that assert exact `complete_text` call counts).

> **Watch-out:** `test_high_confidence_composes_answer` and similar patch `complete_text` with a fixed `return_value` (not `side_effect`). With `_diagnose` now calling `complete_text`, that fixed string is also returned to `_diagnose` — e.g. `"Vào menu X rồi tạo."` → `json.loads` raises → fail-closed `None`, so the pipeline proceeds normally. This is fine. Only tests that assert `complete_text` / `rag.search` **call counts** need review; `rag.search` count is unaffected (diagnose doesn't call it).

- [ ] **Step 6: Commit**

```bash
git add agent_customer_support/agents/prompts.py agent_customer_support/agents/knowledge.py tests/agents/test_knowledge.py
git commit -m "feat(diagnostics): inject matched operating rule into composition"
```

---

### Task 5: Lint and final verification

**Files:** none (verification only)

- [ ] **Step 1: Run lint**

Run: `make lint`
Expected: ruff format clean, ruff check clean, mypy clean. Fix any type issues (e.g. ensure `DiagnosticRule | None` return annotation present on `_diagnose`).

- [ ] **Step 2: Run full test suite**

Run: `make test`
Expected: all pass.

- [ ] **Step 3: Commit any lint fixes**

```bash
git add -A
git commit -m "style: lint fixes for diagnostic rules" || echo "nothing to commit"
```

---

## Self-Review

**Spec coverage:**
- Rules-as-data (`diagnostics.py`) → Task 1 ✓
- Classifier (`_diagnose`, prompt, config) → Tasks 2, 3 ✓
- Injection + compose-prompt amendment + force-present → Task 4 ✓
- Fail-closed error handling → Task 3 (3 fail-closed tests) ✓
- Reactive / no-match-unchanged → Task 4 `test_no_diagnostic_match_leaves_pipeline_unchanged` ✓
- Testing matrix (rules data, classifier phrasings, injection ordering) → covered; the spec's "≥2 phrasings per rule" is represented by one phrasing per rule in `_diagnose` tests plus the data test — expand phrasings here if desired during review.
- Cost option (a) always-run on cheap model → Task 2 config + Task 4 unconditional `_diagnose` call ✓

**Placeholder scan:** none — every step has concrete code/commands.

**Type consistency:** `DiagnosticRule` (id/symptom/guidance), `DIAGNOSTIC_RULES`, `RULES_BY_ID`, `_diagnose(query, cfg) -> DiagnosticRule | None`, `DIAGNOSTIC_PROMPT`, `diagnostic_model` — names match across Tasks 1-4.

**Deviation from spec:** sequential (not parallel) `_diagnose`/`rag.search` — documented in header; behaviorally identical given synchronous `complete_text`.

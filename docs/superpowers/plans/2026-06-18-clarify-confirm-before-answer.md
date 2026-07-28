# Clarify / Confirm / Diagnose Before Answering — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `KnowledgeAgent` two capabilities beyond plain Q&A: **(1) clarify/confirm** — ask one question before answering when a missing/ambiguous element materially changes the answer and can't be resolved from context or retrieval; **(2) diagnose** — when the situation the user *describes* contradicts a process rule, surface the violation and its root cause (grounded in the process) before/instead of just answering the literal question. The two chain: confirm the case → diagnose the mistake → guide the fix.

**Architecture:** Both capabilities live *inside* the existing compose LLM (it already holds both grounding sources — always-on `PROCESS_BLOCK` + RAG passages — and is the only step that can judge answerability and conformance).
- **Clarify/confirm** uses a single `[[clarify]]` marker (ambiguous subject, missing decisive parameter, unverified premise, high-stakes intent). It reuses the existing one-shot loop (`session.pending = "knowledge_clarify"`); the resume turn re-runs the full pipeline (contextualize → search → compose) so the answer is grounded fresh, not paraphrased from the option list. On resume the composer is told not to clarify again, bounding the loop to one round-trip.
- **Diagnose** needs **no marker and no `run()` change** — it is normal grounded answer text (`resolved=True`). It is purely a compose-prompt instruction: detect a contradiction between the user's described actions and a process rule, explain the root cause, then guide. Because it has no mechanical signal, its correctness is validated by **behavioral eval**, not unit tests.

The PQT sample (state-branch) exercises clarify; the "PYC created 18/4 before sample received 20/4 in a B7-B case" sample exercises the **chain**: confirm B7-B → diagnose the date violation (B6: *"chỉ tạo đơn khi mẫu đã đến công ty"*) → guide via "công việc không phù hợp".

**Tech Stack:** Python 3 / Pytest / async. Single LLM facade (`complete_text`). No new model, no Coordinator change.

---

## Core invariant (applies to every clarify/confirm case)

Clarify/confirm only to obtain **user-state or intent the agent cannot know or verify**. Never fabricate the answer. The final answer must be grounded in the two sources. Anything enumerated (options, premises) must also be grounded — no invented branches.

**Clarify when** a missing/ambiguous element (a) *materially changes* the answer AND (b) cannot be resolved from conversation/screenshot or by retrieving the sources. **Default: answer.**

Typical triggers (guidance for the prompt, not rigid code):
- **Ambiguous subject** — the question maps to several distinct features/objects whose answers differ (e.g. "tạo phiếu" → báo giá vs PYC vs phiếu kết quả).
- **Missing decisive parameter** — answer depends on the order's current step, the user's role/department, or which application — none visible to the agent.
- **Unverified premise** — the question assumes something happened/holds that may not (e.g. "khi PQT trả đơn về thì…").
- **High-stakes / irreversible intent** — guidance touches destructive or hard-to-reverse steps (huỷ PYC; re-issue after BGĐ ký).

**Do NOT clarify** when: only one reading is plausible in context; all branches converge on the same answer; or the missing piece is process knowledge the agent should retrieve (don't push lookup onto the user).

### Diagnose (process-conformance) — distinct from clarify

Clarify gathers *missing* info. **Diagnose** detects a *contradiction*: the situation the user describes violates a process rule, so the literally-correct answer to their question would leave them satisfied-but-wrong. The agent must surface the violation and its root cause (grounded in the process) before guiding the fix.

**Diagnose when** the user's described actions/state contradict a stated process rule, AND that contradiction is material to their question. Then: name the mistake, cite the rule plainly ("theo quy trình…"), explain the consequence, and only then give the corrective guidance.

Example (the B7-B sample): user created PYC 18/4 but the sample arrived 20/4 in a nhận-mẫu-tại-cty case. Rule B6: *"chỉ tạo đơn khi mẫu đã đến công ty"* → PYC date should equal receipt date. The literal question ("can I edit the receipt date?") has a correct answer (công việc không phù hợp), but answering only that hides the real error. The agent should: confirm it's B7-B → point out "tạo PYC trước ngày nhận mẫu là sai quy trình" → then guide.

**Do NOT diagnose** (over-correction risk, same high bar as clarify) when: the contradiction is speculative/not grounded in a specific rule; or it's immaterial to what the user asked. Default is to answer, not to lecture.

## Why detection is in the composer, not a new agent

- The composer is the only step holding *both* grounding sources at once. A separate clarify-classifier would re-derive that context and could disagree with the composer about answerability.
- Deciding to clarify is inseparable from grounding the answer — both bound to the same two sources and the same anti-fabrication rules.
- No second model call → no added latency/cost.

The Coordinator already routes a `knowledge_clarify` pending turn straight back to `KnowledgeAgent` (bypassing Triage, `coordinator.py:121`), and `_knowledge_phase` returns a `resolved is None` result as-is. So **no Coordinator edits are needed** — only `prompts.py`, `knowledge.py`, and tests.

## File Structure

- Modify: `agent_customer_support/agents/prompts.py` — add the general clarify/confirm policy to `KNOWLEDGE_COMPOSE_PROMPT`; add `KNOWLEDGE_RESUME_NO_CLARIFY` constant.
- Modify: `agent_customer_support/agents/knowledge.py` — add `_CLARIFY_RE`, handle `clarify` kind in `parse_markers`, thread `allow_clarify` through `_compose`, branch in `run()`.
- Test: `tests/agents/test_knowledge.py` — parse-marker unit, clarify pipeline branch (multiple trigger categories), bounded-loop / re-ground behavior, compose directive injection.

---

### Task 1: `[[clarify]]` marker parsing

**Files:**
- Modify: `agent_customer_support/agents/knowledge.py:18-35`
- Test: `tests/agents/test_knowledge.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/agents/test_knowledge.py` (near the other `parse_markers` tests):

```python
def test_parse_markers_clarify():
    clean, kind, mod = parse_markers(
        "Bạn đang muốn tạo loại phiếu nào?\n- Báo giá\n- PYC\n- Phiếu kết quả [[clarify]]"
    )
    assert kind == "clarify"
    assert mod is None
    assert "[[clarify]]" not in clean
    assert "Báo giá" in clean  # grounded options survive


def test_parse_markers_bug_beats_clarify():
    # If the model emits both, suspected_bug wins (safe handoff path).
    clean, kind, mod = parse_markers("Đáng lẽ chạy. [[clarify]] [[suspected_bug:xn]]")
    assert kind == "suspected_bug" and mod == "xn"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/agents/test_knowledge.py::test_parse_markers_clarify tests/agents/test_knowledge.py::test_parse_markers_bug_beats_clarify -v`
Expected: FAIL — `parse_markers` returns `kind=None` (clarify marker unrecognized; marker text leaks into `clean`).

- [ ] **Step 3: Add the regex and parse branch**

In `agent_customer_support/agents/knowledge.py`, add the regex after `_BUG_RE`:

```python
_NO_ANSWER_RE = re.compile(r"\[\[no_answer\]\]")
_BUG_RE = re.compile(r"\[\[suspected_bug:([a-zA-Z0-9_\-]+)\]\]")
_CLARIFY_RE = re.compile(r"\[\[clarify\]\]")
```

Replace the `parse_markers` body so clarify is checked after bug, before no_answer:

```python
def parse_markers(text: str) -> tuple[str, str | None, str | None]:
    """Return (clean_text, kind, application) where kind in
    {None, 'no_answer', 'suspected_bug', 'clarify'}.

    Precedence: suspected_bug > clarify > no_answer. A bug is the safest handoff,
    so it wins if the model emits more than one marker.
    """
    bug = _BUG_RE.search(text or "")
    if bug:
        clean = _BUG_RE.sub("", text).strip()
        return clean, "suspected_bug", bug.group(1)
    if _CLARIFY_RE.search(text or ""):
        clean = _CLARIFY_RE.sub("", text).strip()
        return clean, "clarify", None
    if _NO_ANSWER_RE.search(text or ""):
        clean = _NO_ANSWER_RE.sub("", text).strip()
        # If the model wrote substantial content AND appended [[no_answer]], the marker
        # is a spurious hedge — trust the content and treat it as a valid answer.
        if len(clean) > 80:
            return clean, None, None
        return clean, "no_answer", None
    return (text or "").strip(), None, None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/agents/test_knowledge.py -k "parse_markers" -v`
Expected: PASS (all `parse_markers` tests, including the two new ones).

- [ ] **Step 5: Commit**

```bash
git add agent_customer_support/agents/knowledge.py tests/agents/test_knowledge.py
git commit -m "feat(knowledge): parse [[clarify]] marker"
```

---

### Task 2: Compose prompt — clarify/confirm policy + diagnose policy + resume suppressor

**Files:**
- Modify: `agent_customer_support/agents/prompts.py:98-118`
- Test: `tests/agents/test_knowledge.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/agents/test_knowledge.py`:

```python
def test_compose_prompt_documents_clarify_and_diagnose_policy():
    from agent_customer_support.agents.prompts import (
        KNOWLEDGE_COMPOSE_PROMPT,
        KNOWLEDGE_RESUME_NO_CLARIFY,
    )

    # The clarify/confirm contract must be in the compose system prompt...
    assert "[[clarify]]" in KNOWLEDGE_COMPOSE_PROMPT
    # ...the diagnose (process-conformance) contract too...
    assert "sai quy trình" in KNOWLEDGE_COMPOSE_PROMPT
    # ...and the resume suppressor must forbid re-clarifying.
    assert "[[clarify]]" in KNOWLEDGE_RESUME_NO_CLARIFY
    assert "KHÔNG" in KNOWLEDGE_RESUME_NO_CLARIFY
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/agents/test_knowledge.py::test_compose_prompt_documents_clarify_and_diagnose_policy -v`
Expected: FAIL — `ImportError: cannot import name 'KNOWLEDGE_RESUME_NO_CLARIFY'`.

- [ ] **Step 3: Edit the compose prompt and add the constant**

In `agent_customer_support/agents/prompts.py`, insert this policy block into `KNOWLEDGE_COMPOSE_PROMPT` immediately before the `CHỐNG BỊA:` line:

```
HỎI LẠI / XÁC NHẬN TRƯỚC KHI TRẢ LỜI: Mặc định trả lời thẳng. CHỈ hỏi lại hoặc xác nhận khi thiếu một yếu tố mà (a) LÀM THAY ĐỔI hẳn câu trả lời, VÀ (b) bạn KHÔNG thể tự suy ra từ hội thoại/ảnh, cũng KHÔNG tra được trong hai nguồn. Hỏi NGẮN, mỗi lượt chỉ hỏi điều cần nhất. Các tình huống điển hình:
- Mơ hồ đối tượng: câu hỏi khớp nhiều chức năng/đối tượng khác nhau và câu trả lời mỗi cái một khác (vd "tạo phiếu" có thể là báo giá / PYC / phiếu kết quả) → hỏi rõ đang nói đến cái nào.
- Thiếu dữ kiện quyết định: câu trả lời phụ thuộc trạng thái/vai trò/ứng dụng mà bạn không thấy (đơn đang ở bước nào, bạn thuộc bộ phận nào, đang ở ứng dụng nào) → hỏi dữ kiện đó. Nếu liệt kê các nhánh, MỖI nhánh phải bám hai nguồn, KHÔNG bịa nhánh.
- Tiền đề chưa chắc: câu hỏi giả định một việc đã xảy ra/đúng nhưng chưa chắc (vd "khi X trả đơn về thì...") → xác nhận tiền đề, hoặc trả lời kèm điều kiện rõ ràng.
KHÔNG hỏi khi: chỉ một cách hiểu hợp lý theo ngữ cảnh; mọi nhánh đều ra cùng kết luận; hoặc thứ còn thiếu là kiến thức quy trình mà bạn tự tra được (đừng đẩy việc tra cứu sang user).
Khi cần hỏi/xác nhận → viết câu hỏi (kèm các lựa chọn CÓ CĂN CỨ nếu có) rồi kết thúc bằng [[clarify]].
```

Insert this diagnosis block immediately after the clarify policy block above (also before `CHỐNG BỊA:`):

```
ĐỐI CHIẾU QUY TRÌNH (chẩn đoán): trước khi trả lời, kiểm tra xem TÌNH HUỐNG user MÔ TẢ có MÂU THUẪN với một quy tắc/điều kiện cụ thể trong quy trình không. Nếu có và điều đó liên quan tới câu hỏi: ĐỪNG chỉ trả lời đúng theo chữ câu hỏi — phải CHỈ RÕ user đang làm sai quy trình ở đâu, dẫn quy tắc một cách tự nhiên ("theo quy trình..."), nêu hệ quả, RỒI mới hướng dẫn cách xử lý đúng.
Ví dụ: case nhận mẫu tại công ty (B7-B) phải tạo PYC ĐÚNG ngày nhận mẫu thực tế; nếu user tạo PYC TRƯỚC ngày nhận mẫu thì đó là sai quy trình — nêu rõ điểm sai trước, rồi mới hướng dẫn.
CHỈ chẩn đoán khi mâu thuẫn CÓ CĂN CỨ rõ ràng trong quy trình và LIÊN QUAN câu hỏi; nếu không, trả lời bình thường, không suy diễn lỗi, không lên lớp.
```

Then replace the `MARKER (tối đa một):` block with:

```
MARKER (tối đa một):
- Cần hỏi lại/xác nhận trước khi trả lời (xem mục HỎI LẠI / XÁC NHẬN) → viết câu hỏi/lựa chọn có căn cứ rồi kết thúc bằng [[clarify]]
- CẢ hai nguồn đều không trả lời được → đúng một dòng: [[no_answer]]
- Tài liệu xác nhận tính năng đáng lẽ chạy nhưng user báo lỗi → kết thúc bằng [[suspected_bug:<application>]]
- Còn lại → trả lời trực tiếp, không kèm marker.
```

Add this constant directly after `KNOWLEDGE_COMPOSE_PROMPT` (after its closing `"""`):

```python
# Appended to the compose user-content on a clarify resume turn (the user is answering
# our earlier clarify/confirm question). Forces a grounded answer instead of a second
# clarify, keeping the loop bounded to one round-trip.
KNOWLEDGE_RESUME_NO_CLARIFY = (
    "LƯU Ý: user vừa trả lời câu hỏi làm rõ/xác nhận trước đó. TUYỆT ĐỐI KHÔNG hỏi lại nữa "
    "(không dùng [[clarify]]). Nếu vẫn còn nhiều khả năng, hãy chọn khả năng hợp lý nhất theo "
    "ngữ cảnh, trả lời và NÊU RÕ giả định/điều kiện đang áp dụng."
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/agents/test_knowledge.py::test_compose_prompt_documents_clarify_and_diagnose_policy -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent_customer_support/agents/prompts.py tests/agents/test_knowledge.py
git commit -m "feat(prompts): clarify/confirm + process-conformance diagnose policy in composer"
```

---

### Task 3: Thread `allow_clarify` through `_compose`

**Files:**
- Modify: `agent_customer_support/agents/knowledge.py:4-9` (imports), `:82-103` (`_compose`)
- Test: `tests/agents/test_knowledge.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/agents/test_knowledge.py`:

```python
async def test_compose_appends_no_clarify_directive_when_disabled():
    from agent_customer_support.agents.prompts import KNOWLEDGE_RESUME_NO_CLARIFY

    agent = KnowledgeAgent()
    captured: dict = {}

    def fake_complete(*, messages, system, model=None):
        captured["content"] = messages[0]["content"]
        return "ok"

    with patch("agent_customer_support.agents.knowledge.complete_text", side_effect=fake_complete):
        await agent._compose("q", ["p"], "user: q", get_settings(), allow_clarify=False)

    assert KNOWLEDGE_RESUME_NO_CLARIFY in captured["content"]


async def test_compose_omits_no_clarify_directive_by_default():
    from agent_customer_support.agents.prompts import KNOWLEDGE_RESUME_NO_CLARIFY

    agent = KnowledgeAgent()
    captured: dict = {}

    def fake_complete(*, messages, system, model=None):
        captured["content"] = messages[0]["content"]
        return "ok"

    with patch("agent_customer_support.agents.knowledge.complete_text", side_effect=fake_complete):
        await agent._compose("q", ["p"], "user: q", get_settings())

    assert KNOWLEDGE_RESUME_NO_CLARIFY not in captured["content"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/agents/test_knowledge.py -k "no_clarify_directive" -v`
Expected: FAIL — `_compose() got an unexpected keyword argument 'allow_clarify'`.

- [ ] **Step 3: Add the import and the parameter**

In `agent_customer_support/agents/knowledge.py`, extend the prompts import block:

```python
from agent_customer_support.agents.prompts import (
    KNOWLEDGE_CONTEXTUALIZE_PROMPT,
    KNOWLEDGE_CONTEXTUALIZE_VISION_PROMPT,
    KNOWLEDGE_COMPOSE_PROMPT,
    KNOWLEDGE_RESUME_NO_CLARIFY,
    PROCESS_BLOCK,
)
```

Change the `_compose` signature. Replace:

```python
    async def _compose(
        self, question: str, passages: list[str], transcript: str, cfg: Settings
    ) -> str:
```

with:

```python
    async def _compose(
        self,
        question: str,
        passages: list[str],
        transcript: str,
        cfg: Settings,
        allow_clarify: bool = True,
    ) -> str:
```

And replace the `content = (...)` assignment with:

```python
        content = (
            f"{history}Câu hỏi hiện tại: {question}\n\nĐoạn trích:\n{_passages_block(passages)}"
        )
        if not allow_clarify:
            content = f"{content}\n\n{KNOWLEDGE_RESUME_NO_CLARIFY}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/agents/test_knowledge.py -k "no_clarify_directive or compose" -v`
Expected: PASS (new directive tests + existing `_compose` tests still green).

- [ ] **Step 5: Commit**

```bash
git add agent_customer_support/agents/knowledge.py tests/agents/test_knowledge.py
git commit -m "feat(knowledge): add allow_clarify flag to _compose for resume turns"
```

---

### Task 4: `run()` clarify branch + bounded loop

**Files:**
- Modify: `agent_customer_support/agents/knowledge.py:105-159`
- Test: `tests/agents/test_knowledge.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/agents/test_knowledge.py`. The first three cover different *trigger categories* but all exercise the same code path (compose → `[[clarify]]` → pending set), documenting that the mechanism is general:

```python
import pytest as _pytest


@_pytest.mark.parametrize(
    "message, clarify_reply",
    [
        # ambiguous subject
        (
            "cách tạo phiếu?",
            "Bạn muốn tạo loại phiếu nào?\n- Báo giá\n- PYC\n- Phiếu kết quả [[clarify]]",
        ),
        # missing decisive parameter / unknown user-state
        (
            "PQT trả đơn về thì KD sửa số lượng mẫu được không?",
            "Đơn của bạn đang ở trạng thái nào?\n- Còn trong ứng dụng\n"
            "- Đã chuyển chưa tiếp nhận\n- Đã tiếp nhận ở ứng dụng khác [[clarify]]",
        ),
        # unverified premise
        (
            "sau khi huỷ PYC thì hoàn tiền thế nào?",
            "Bạn đã thực sự huỷ PYC chưa, hay đang cân nhắc? [[clarify]]",
        ),
    ],
)
async def test_clarify_asks_once_and_sets_pending(message, clarify_reply):
    ctx = _ctx(message)
    ctx.rag.search.return_value = {"passages": [], "citations": ["c#1"]}
    with patch(
        "agent_customer_support.agents.knowledge.complete_text", return_value=clarify_reply
    ):
        res = await KnowledgeAgent().run(ctx)
    assert res.resolved is None  # neither answered nor escalated
    assert ctx.session.pending == "knowledge_clarify"
    assert "[[clarify]]" not in res.reply
    ctx.backlog.add.assert_not_awaited()  # clarify is not a miss


async def test_resume_turn_disables_clarify_and_grounds_answer():
    """On resume, compose is called with allow_clarify=False; a grounded answer returns."""
    ctx = _ctx("đơn đang còn trong ứng dụng")
    ctx.session.pending = "knowledge_clarify"  # we clarified last turn
    ctx.rag.search.return_value = {"passages": ["p" * 200], "citations": []}
    seen: dict = {}

    def fake_complete(**kwargs):
        content = kwargs["messages"][0]["content"]
        if "Đoạn trích" in content:  # the compose call
            seen["compose_content"] = content
            return "Vì đơn còn trong ứng dụng, bạn trả về tài khoản đã tạo để sửa."
        return kwargs["messages"][0]["content"]  # contextualize passthrough

    with patch(
        "agent_customer_support.agents.knowledge.complete_text", side_effect=fake_complete
    ):
        res = await KnowledgeAgent().run(ctx)

    from agent_customer_support.agents.prompts import KNOWLEDGE_RESUME_NO_CLARIFY

    assert res.resolved is True
    assert ctx.session.pending is None  # flag consumed
    assert KNOWLEDGE_RESUME_NO_CLARIFY in seen["compose_content"]  # clarify suppressed
    ctx.backlog.add.assert_not_awaited()


async def test_clarify_marker_on_resume_is_downgraded_to_answer():
    """Defensive: if the model disobeys and re-emits [[clarify]] on resume, answer anyway."""
    ctx = _ctx("vẫn chưa rõ")
    ctx.session.pending = "knowledge_clarify"
    ctx.rag.search.return_value = {"passages": [], "citations": []}
    with patch(
        "agent_customer_support.agents.knowledge.complete_text",
        return_value="Giả định đơn còn trong ứng dụng: bạn sửa trực tiếp. [[clarify]]",
    ):
        res = await KnowledgeAgent().run(ctx)
    assert res.resolved is True  # not a second clarify
    assert ctx.session.pending is None
    assert "[[clarify]]" not in res.reply
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/agents/test_knowledge.py -k "clarify_asks_once or resume_turn_disables or downgraded" -v`
Expected: FAIL — without the clarify branch, a `[[clarify]]` reply falls into the plain-answer path (`resolved is True`, `pending` never set), and `_compose` is called with `allow_clarify=True` on resume.

- [ ] **Step 3: Wire the clarify branch into `run()`**

In `agent_customer_support/agents/knowledge.py`, in `run()`, change the compose call to pass `allow_clarify`:

```python
        composed = await self._compose(
            query, passages, ctx.transcript, cfg, allow_clarify=not already_clarified
        )
        clean, kind, application = parse_markers(composed)
```

Then insert the clarify branch immediately after the `suspected_bug` block and **before** the `if kind != "no_answer":` line:

```python
        if kind == "suspected_bug":
            return AgentResult(
                reply=clean,
                resolved=False,
                suspected_bug=True,
                evidence={"application": application, "summary": ctx.message},
                citations=citations,
            )

        # Clarify / confirm before answering. The composer judged that an element it
        # can't see (ambiguous subject, unknown user-state, unverified premise, or a
        # risky intent) materially changes the answer. Ask once — bounded by the same
        # knowledge_clarify flag — then re-ground on the user's reply next turn.
        # allow_clarify=False on the resume turn means compose should never reach here
        # twice; if the model disobeys, downgrade to a plain (assumption-stated) answer.
        if kind == "clarify":
            if not already_clarified:
                ctx.session.pending = "knowledge_clarify"
                return AgentResult(reply=clean, resolved=None, citations=citations)
            return AgentResult(reply=clean, resolved=True, citations=citations)

        if kind != "no_answer":
            return AgentResult(reply=clean, resolved=True, citations=citations)
```

(The `no_answer` first-miss/second-miss logic below stays unchanged. Note: the existing vague-question clarify on `[[no_answer]]` still works and is also bounded by `already_clarified`, so the two clarify paths never stack.)

- [ ] **Step 4: Run the full knowledge suite**

Run: `poetry run pytest tests/agents/test_knowledge.py -v`
Expected: PASS (all new tests + every pre-existing test).

- [ ] **Step 5: Commit**

```bash
git add agent_customer_support/agents/knowledge.py tests/agents/test_knowledge.py
git commit -m "feat(knowledge): clarify/confirm before answering, bounded to one turn"
```

---

### Task 5: Full regression + lint

**Files:** none (verification only)

- [ ] **Step 1: Run the whole test suite**

Run: `make test`
Expected: PASS. Pay attention to `tests/agents/test_coordinator.py` — confirm the `knowledge_clarify` resume path (already routed in `coordinator.py:121`) still returns a `resolved is None` clarify reply unchanged.

- [ ] **Step 2: Lint**

Run: `make lint`
Expected: ruff format clean, ruff check clean, mypy clean. `_compose`'s new keyword arg is typed `bool`; `parse_markers` return type unchanged (`tuple[str, str | None, str | None]`).

- [ ] **Step 3: Commit any formatting**

```bash
git add -A
git commit -m "style: ruff format clarify/confirm changes"
```

---

### Task 6: Behavioral eval for clarify / diagnose

Unit tests only prove the mechanism *fires when the marker is emitted*. Whether the policy fires *at the right times* — clarifies a genuinely ambiguous question, diagnoses a real violation, and stays quiet otherwise — is only measurable behaviorally (real LLM, no mock). The hallucination judge's verdicts (GROUNDED/HALLUCINATED/REFUSAL) don't fit, so this is a **separate** eval script with its own judge labels.

**Files:**
- Create: `scripts/eval_clarify_diagnose.py`

- [ ] **Step 1: Create the eval script**

Model it on `scripts/eval_hallucination.py` (same Coordinator wiring + colour/summary scaffolding). The judge classifies the agent's *behavior* against an expected mode:

```python
"""
Clarify / Diagnose behavioral evaluation.

Runs the agent on cases that SHOULD trigger a clarify, a diagnosis, or neither,
and uses an LLM judge to classify the reply's behavior. Validates the policy
fires at the right times (and not when it shouldn't).

Usage:
    set -a && source .env && set +a
    poetry run python scripts/eval_clarify_diagnose.py
"""
import asyncio
import json
import uuid
from dataclasses import dataclass

from agent_customer_support.models import CustomerProfile
from agent_customer_support.stores.customer_registry import CustomerRegistry
from agent_customer_support.agents.coordinator import Coordinator
from agent_customer_support.llm import complete_with_tools


@dataclass
class Case:
    id: str
    turns: list[str]            # one or more user turns (multi-turn for the chain)
    expected: str              # CLARIFY | DIAGNOSE | DIRECT_ANSWER
    note: str


CASES: list[Case] = [
    # State-branch → must ask which state before answering.
    Case(
        id="pqt_return_branch",
        turns=["Đơn được PQT trả về cho KD thì KD sửa số lượng mẫu trong đơn đã tạo được không?"],
        expected="CLARIFY",
        note="answer forks on order state the agent can't see",
    ),
    # Ambiguous subject → must ask which 'phiếu'.
    Case(
        id="ambiguous_phieu",
        turns=["cách tạo phiếu?"],
        expected="CLARIFY",
        note="'phiếu' = báo giá / PYC / phiếu kết quả",
    ),
    # The B7-B chain: turn 1 should confirm the case; turn 2 (user confirms) should
    # DIAGNOSE the date violation, not just explain how to edit the date.
    Case(
        id="b7b_date_violation",
        turns=[
            "Em tạo PYC ngày 18/4/2026 nhưng ngày nhận mẫu là 20/4/2026, sửa lại "
            "ngày nhận mẫu cho khớp được không ạ?",
            "Đúng rồi, đây là trường hợp khách đem mẫu tới công ty.",
        ],
        expected="DIAGNOSE",
        note="created PYC before receipt in B7-B = process violation; must point it out",
    ),
    # Negative control → clear question, must NOT clarify/diagnose.
    Case(
        id="direct_who_approves",
        turns=["Ai phụ trách bước nghiệm thu hợp đồng?"],
        expected="DIRECT_ANSWER",
        note="unambiguous process question",
    ),
]

JUDGE_SYSTEM = """Bạn phân loại HÀNH VI của câu trả lời Agent (không chấm đúng/sai nội dung).
Nhãn:
- CLARIFY: Agent hỏi lại/xác nhận để lấy thông tin còn thiếu trước khi trả lời.
- DIAGNOSE: Agent chỉ ra user đang làm SAI QUY TRÌNH (nêu điểm sai + lý do) rồi mới hướng dẫn.
- DIRECT_ANSWER: Agent trả lời thẳng, không hỏi lại, không chỉ ra lỗi quy trình.
Trả về JSON: {"verdict":"<nhãn>","reason":"<1 câu>"}"""


def judge(question: str, reply: str) -> dict:
    out = complete_with_tools(
        messages=[{"role": "user", "content": f"Hội thoại:\n{question}\n\nCâu trả lời cuối của Agent:\n{reply}"}],
        tools=[],
        system=JUDGE_SYSTEM,
    )
    raw = (out.get("text") or "").strip()
    if "```" in raw:
        raw = raw.split("```")[1].removeprefix("json")
    try:
        return json.loads(raw)
    except Exception:
        return {"verdict": "PARSE_ERROR", "reason": raw[:120]}


async def run_eval() -> None:
    reg = CustomerRegistry()
    await reg.init()
    await reg.put(CustomerProfile(
        customer_id="eval_user", name="EvalUser",
        enabled_applications=["yeu-cau-thu-nghiem", "lay-mau-quan-trac"],
    ))
    agent = Coordinator()
    passed = 0
    for case in CASES:
        conv_id = f"clarify-{case.id}-{uuid.uuid4().hex[:6]}"
        reply = ""
        joined = ""
        for turn in case.turns:
            joined += f"user: {turn}\n"
            resp = await agent.handle_turn(
                customer_id="eval_user", conversation_id=conv_id,
                message=turn, attachments=[],
            )
            reply = resp.reply
            joined += f"assistant: {reply}\n"
        verdict = judge(joined, reply)
        ok = verdict.get("verdict") == case.expected
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {case.id}: want {case.expected}, "
              f"got {verdict.get('verdict')} — {verdict.get('reason','')}")
    print(f"\nScore: {passed}/{len(CASES)}")


if __name__ == "__main__":
    asyncio.run(run_eval())
```

- [ ] **Step 2: Run the eval against the live agent**

Run:
```bash
set -a && source .env && set +a
poetry run python scripts/eval_clarify_diagnose.py
```
Expected: a per-case PASS/FAIL table. `pqt_return_branch` and `ambiguous_phieu` → CLARIFY; `b7b_date_violation` → DIAGNOSE; `direct_who_approves` → DIRECT_ANSWER. Treat failures as **prompt-tuning signal** (adjust the Task 2 policy wording and re-run) — this is the loop where the when-to-clarify / when-to-diagnose thresholds actually get calibrated. Do not loosen the bar so far that the negative control starts clarifying.

- [ ] **Step 3: Commit**

```bash
git add scripts/eval_clarify_diagnose.py
git commit -m "test(eval): behavioral eval for clarify/diagnose policy"
```

---

## Self-Review

**Spec coverage (answer + clarify/confirm + diagnose):**
- General clarify/confirm decision rule → Task 2 policy block (4 trigger categories + non-triggers + the materially-changes + can't-self-resolve bar). ✅
- Single mechanism for clarify *and* confirm → one `[[clarify]]` marker (Task 1) + `run()` branch (Task 4). ✅
- **Diagnose (process-conformance)** → Task 2 diagnosis block (prompt-only; no marker/`run()` change); validated by Task 6 `b7b_date_violation`. ✅
- Never fabricate; final answer grounded; re-ground after reply → Task 4 resume re-runs the full pipeline; `test_resume_turn_disables_clarify_and_grounds_answer` asserts the fresh compose call. ✅
- Bounded to one turn → `allow_clarify=not already_clarified` + downgrade test. ✅
- Don't ask what's retrievable / over-correct → prompt rules in Task 2; guarded by Task 6 negative control (`direct_who_approves`). ✅

**Placeholder scan:** none — every code/test step is complete; the eval script is full, not stubbed.

**Type consistency:** `_compose(..., allow_clarify: bool = True)` matches the call in `run()`; `parse_markers` kind values `{None,'no_answer','suspected_bug','clarify'}` match the `run()` branches; `KNOWLEDGE_RESUME_NO_CLARIFY` defined in Task 2, imported in Task 3, asserted in Tasks 3 & 4. Diagnose adds no new types (it's answer text).

## Notes / follow-ups

- **Diagnosis is prompt-based, not rule-based.** It relies on the composer reasoning over `PROCESS_CONTEXT`. If Task 6 shows it unreliable on high-value violations, consider reviving a small structured layer (the deleted `agent_customer_support/agents/diagnostics.py` had a `DiagnosticRule` symptom→guidance pattern) for those specific cases, while keeping the prompt path as the general fallback.
- **Resume routing for diagnosis chain:** the B7-B chain works because the confirm turn sets `knowledge_clarify`, and the resume turn (user confirms B7-B) re-composes with the confirmation in the transcript — so the composer diagnoses on the second turn. No extra wiring; confirm reuses the clarify loop.

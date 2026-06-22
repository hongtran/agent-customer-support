# Q&A Learning Loop 2b — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `KnowledgeAgent` answer from the curated `qa` collection alongside the product guide, with confidence-gated precedence (a strongly-matching CS answer leads; otherwise it's supplementary and the guide leads).

**Architecture:** A second `RagClient.search` against the `qa` collection (reusing the Spec-1 read path) feeds `_compose` together with the guide passages plus a `qa_leads` flag. Composition uses the unchanged two-source prompt when no Q&A matched (zero regression), and a three-source variant when Q&A is present. The qa search degrades to product-only on any error.

**Tech Stack:** Python 3.13, existing `RagClient`/`complete_text`, pytest + pytest-asyncio.

## Global Constraints

- `qa_leads = bool(qa_passages) and qa_res["top_confidence"] >= qa_lead_threshold` (default `0.85`).
- Three-tier precedence on conflict: (1) CS answer marked "ưu tiên cao nhất" when it leads > (2) QUY TRÌNH > (3) ĐOẠN TRÍCH. Non-leading Q&A is "bổ trợ" and does NOT outrank QUY TRÌNH.
- When there are **no** qa passages, `_compose` must behave byte-identically to today: existing `KNOWLEDGE_COMPOSE_PROMPT`, no CS block in the user content.
- The qa search degrades to product-only (empty result) on ANY exception — the guide path must never break because Q&A is unavailable (the `qa` collection does not exist until Spec-2a's first approval).
- The qa search uses the SAME `applications` filter as the product search.
- qa citations are merged into the result prefixed `qa:` (e.g. `qa:<record_id>`).
- The CS-block headers in the compose content must contain the exact tokens the prompt keys off: `ưu tiên cao nhất` (leading) / `bổ trợ` (supplementary).

---

### Task 1: Config knob + three-source compose prompt

**Files:**
- Modify: `agent_customer_support/config.py`
- Modify: `agent_customer_support/agents/prompts.py`
- Test: `tests/agents/test_prompts_qa.py` (create), `tests/test_config_qa_2b.py` (create)

**Interfaces:**
- Produces: `Settings.qa_lead_threshold: float` (default 0.85); `agent_customer_support.agents.prompts.KNOWLEDGE_COMPOSE_PROMPT_WITH_QA: str`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_config_qa_2b.py`:

```python
from agent_customer_support.config import Settings


def test_qa_lead_threshold_default():
    assert Settings().qa_lead_threshold == 0.85
```

Create `tests/agents/test_prompts_qa.py`:

```python
from agent_customer_support.agents.prompts import (
    KNOWLEDGE_COMPOSE_PROMPT,
    KNOWLEDGE_COMPOSE_PROMPT_WITH_QA,
)


def test_with_qa_prompt_is_three_source_variant():
    p = KNOWLEDGE_COMPOSE_PROMPT_WITH_QA
    # third source introduced
    assert "ba nguồn" in p
    assert "ĐÁP ÁN CS XÁC NHẬN" in p
    # three-tier precedence tokens present
    assert "ưu tiên cao nhất" in p
    assert "bổ trợ" in p
    # miss marker updated to all sources
    assert "Tất cả các nguồn" in p
    # anti-hallucination updated
    assert "ngoài ba nguồn" in p
    # two-source phrasing must NOT remain in the variant
    assert "hai nguồn" not in p


def test_base_prompt_unchanged_is_two_source():
    # the original prompt stays two-source (no regression to the default path)
    assert "hai nguồn" in KNOWLEDGE_COMPOSE_PROMPT
    assert "ĐÁP ÁN CS XÁC NHẬN" not in KNOWLEDGE_COMPOSE_PROMPT
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_config_qa_2b.py tests/agents/test_prompts_qa.py -v`
Expected: FAIL (`AttributeError: qa_lead_threshold` / `ImportError: KNOWLEDGE_COMPOSE_PROMPT_WITH_QA`).

- [ ] **Step 3: Add the config field**

In `agent_customer_support/config.py`, add to the `# RAG (Qdrant read path)` block (after `qa_collection` from Spec 2a, or near the other RAG settings):

```python
    qa_lead_threshold: float = 0.85
```

- [ ] **Step 4: Add the three-source prompt variant**

In `agent_customer_support/agents/prompts.py`, immediately AFTER the existing `KNOWLEDGE_COMPOSE_PROMPT = """..."""` assignment, add a new constant built by copying `KNOWLEDGE_COMPOSE_PROMPT` and applying exactly these five string replacements. Write it out as a full triple-quoted string so it is self-contained:

```python
# Three-source variant of KNOWLEDGE_COMPOSE_PROMPT, used only when CS-verified Q&A
# passages are present. Identical to KNOWLEDGE_COMPOSE_PROMPT except the five deltas
# below (source count, the added source #3, a 3-tier precedence rule, the
# anti-hallucination line, and the [[no_answer]] marker). Keep every other line
# verbatim so the tuned diagnosis/clarify/admin-routing behavior is preserved.
KNOWLEDGE_COMPOSE_PROMPT_WITH_QA = KNOWLEDGE_COMPOSE_PROMPT.replace(
    "Chỉ dựa trên hai nguồn dưới.",
    "Chỉ dựa trên ba nguồn dưới.",
).replace(
    "2. ĐOẠN TRÍCH (passages dưới, RAG lọc theo đúng ứng dụng): chi tiết bên trong MỘT ứng dụng/module — flow nội bộ, logic/nghiệp vụ, thao tác UI. CÓ THỂ RỖNG.",
    "2. ĐOẠN TRÍCH (passages dưới, RAG lọc theo đúng ứng dụng): chi tiết bên trong MỘT ứng dụng/module — flow nội bộ, logic/nghiệp vụ, thao tác UI. CÓ THỂ RỖNG.\n"
    "3. ĐÁP ÁN CS XÁC NHẬN (nếu có, hiển thị dưới đoạn trích): câu trả lời do nhân viên CS biên soạn và duyệt cho đúng câu hỏi này — đã được người thật kiểm chứng. Được đánh dấu \"ưu tiên cao nhất\" hoặc \"bổ trợ\".",
).replace(
    "ƯU TIÊN khi hai nguồn mâu thuẫn: QUY TRÌNH chuẩn cho trình tự liên-module/điều kiện/phân quyền/điểm kiểm soát; ĐOẠN TRÍCH chuẩn cho chi tiết nội bộ module (flow, logic, UI).",
    "ƯU TIÊN khi các nguồn mâu thuẫn (thứ tự giảm dần): (1) ĐÁP ÁN CS XÁC NHẬN đánh dấu \"ưu tiên cao nhất\" — thắng tất cả, kể cả QUY TRÌNH, cho đúng câu hỏi đó; (2) QUY TRÌNH chuẩn cho trình tự liên-module/điều kiện/phân quyền/điểm kiểm soát; (3) ĐOẠN TRÍCH chuẩn cho chi tiết nội bộ module (flow, logic, UI). ĐÁP ÁN CS đánh dấu \"bổ trợ\" chỉ để tham khảo, KHÔNG vượt QUY TRÌNH.",
).replace(
    "Không dùng kiến thức ngoài hai nguồn.",
    "Không dùng kiến thức ngoài ba nguồn.",
).replace(
    "CẢ hai nguồn đều không trả lời được → đúng một dòng: [[no_answer]]",
    "Tất cả các nguồn đều không trả lời được → đúng một dòng: [[no_answer]]",
)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `poetry run pytest tests/test_config_qa_2b.py tests/agents/test_prompts_qa.py -v`
Expected: PASS (3 tests). If `test_with_qa_prompt_is_three_source_variant` fails on `assert "hai nguồn" not in p`, a `.replace(...)` target string did not match the current prompt verbatim — re-copy the exact source line from `prompts.py` into the replace call.

- [ ] **Step 6: Commit**

```bash
git add agent_customer_support/config.py agent_customer_support/agents/prompts.py tests/test_config_qa_2b.py tests/agents/test_prompts_qa.py
git commit -m "feat(qa): qa_lead_threshold + three-source compose prompt variant

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `_compose` — conditional prompt + CS block

**Files:**
- Modify: `agent_customer_support/agents/knowledge.py` (`_compose` + import the new prompt)
- Test: `tests/agents/test_knowledge_compose_qa.py` (create)

**Interfaces:**
- Consumes: `KNOWLEDGE_COMPOSE_PROMPT_WITH_QA` (Task 1).
- Produces: updated `_compose(self, question, passages, transcript, cfg, allow_clarify=True, qa_passages=None, qa_leads=False) -> str`.

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_knowledge_compose_qa.py`:

```python
import pytest
import agent_customer_support.agents.knowledge as kn
from agent_customer_support.agents.knowledge import KnowledgeAgent
from agent_customer_support.agents.prompts import (
    KNOWLEDGE_COMPOSE_PROMPT,
    KNOWLEDGE_COMPOSE_PROMPT_WITH_QA,
)
from agent_customer_support.config import get_settings

pytestmark = pytest.mark.asyncio


def _capture(monkeypatch):
    cap = {}

    def fake_complete_text(messages, system, model):
        cap["content"] = messages[0]["content"]
        cap["system_text"] = system[-1]["text"] if isinstance(system, list) else system
        return "Anh/Chị vui lòng làm theo hướng dẫn."

    monkeypatch.setattr(kn, "complete_text", fake_complete_text)
    return cap


async def test_no_qa_uses_two_source_prompt(monkeypatch):
    cap = _capture(monkeypatch)
    agent = KnowledgeAgent()
    await agent._compose("q", ["guide passage"], "", get_settings())
    assert cap["system_text"] == KNOWLEDGE_COMPOSE_PROMPT
    assert "ĐÁP ÁN CS" not in cap["content"]


async def test_qa_leads_uses_three_source_authoritative_block(monkeypatch):
    cap = _capture(monkeypatch)
    agent = KnowledgeAgent()
    await agent._compose(
        "q", ["guide"], "", get_settings(), qa_passages=["cs answer"], qa_leads=True
    )
    assert cap["system_text"] == KNOWLEDGE_COMPOSE_PROMPT_WITH_QA
    assert "ĐÁP ÁN CS XÁC NHẬN" in cap["content"]
    assert "ưu tiên cao nhất" in cap["content"]
    assert "cs answer" in cap["content"]


async def test_qa_supplementary_uses_three_source_supplementary_block(monkeypatch):
    cap = _capture(monkeypatch)
    agent = KnowledgeAgent()
    await agent._compose(
        "q", ["guide"], "", get_settings(), qa_passages=["cs answer"], qa_leads=False
    )
    assert cap["system_text"] == KNOWLEDGE_COMPOSE_PROMPT_WITH_QA
    assert "bổ trợ" in cap["content"]
    assert "ưu tiên cao nhất" not in cap["content"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/agents/test_knowledge_compose_qa.py -v`
Expected: FAIL (`_compose() got an unexpected keyword argument 'qa_passages'`).

- [ ] **Step 3: Update the import**

In `agent_customer_support/agents/knowledge.py`, extend the prompts import to include the new constant:

```python
from agent_customer_support.agents.prompts import (
    KNOWLEDGE_CONTEXTUALIZE_PROMPT,
    KNOWLEDGE_CONTEXTUALIZE_VISION_PROMPT,
    KNOWLEDGE_COMPOSE_PROMPT,
    KNOWLEDGE_COMPOSE_PROMPT_WITH_QA,
    KNOWLEDGE_RESUME_NO_CLARIFY,
    PROCESS_BLOCK,
)
```

- [ ] **Step 4: Rewrite `_compose`**

Replace the entire `_compose` method body in `agent_customer_support/agents/knowledge.py` with:

```python
    async def _compose(
        self,
        question: str,
        passages: list[str],
        transcript: str,
        cfg: Settings,
        allow_clarify: bool = True,
        qa_passages: list[str] | None = None,
        qa_leads: bool = False,
    ) -> str:
        """Compose a grounded answer from the always-on process + retrieved passages.

        When CS-verified Q&A passages are present, switch to the three-source prompt
        and append a CS-answer block — marked authoritative when qa_leads, else
        supplementary. With no qa_passages, behavior is identical to the two-source
        path (default).
        """
        qa_passages = qa_passages or []
        if _HAS_PRIOR_TURN in transcript:
            history = f"Lịch sử hội thoại:\n{transcript}\n\n"
        else:
            history = ""
        content = (
            f"{history}Câu hỏi hiện tại: {question}\n\nĐoạn trích:\n{_passages_block(passages)}"
        )
        if qa_passages:
            header = (
                "ĐÁP ÁN CS XÁC NHẬN — ưu tiên cao nhất cho câu hỏi này:"
                if qa_leads
                else "ĐÁP ÁN CS XÁC NHẬN — bổ trợ:"
            )
            content = f"{content}\n\n{header}\n{_passages_block(qa_passages)}"
            compose_prompt = KNOWLEDGE_COMPOSE_PROMPT_WITH_QA
        else:
            compose_prompt = KNOWLEDGE_COMPOSE_PROMPT
        if not allow_clarify:
            content = f"{content}\n\n{KNOWLEDGE_RESUME_NO_CLARIFY}"
        return complete_text(
            messages=[{"role": "user", "content": content}],
            system=[PROCESS_BLOCK, {"type": "text", "text": compose_prompt}],
            model=cfg.model_for("knowledge"),
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `poetry run pytest tests/agents/test_knowledge_compose_qa.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add agent_customer_support/agents/knowledge.py tests/agents/test_knowledge_compose_qa.py
git commit -m "feat(qa): _compose conditional 3-source prompt + CS answer block

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Retrieval merge in `KnowledgeAgent.run`

**Files:**
- Modify: `agent_customer_support/agents/knowledge.py` (`run` + new `_safe_qa_search` + logger)
- Test: `tests/agents/test_knowledge_qa_merge.py` (create)

**Interfaces:**
- Consumes: `_compose(..., qa_passages=, qa_leads=)` (Task 2); `Settings.qa_collection` (Spec 2a), `Settings.qa_lead_threshold` (Task 1); `ctx.rag.search(query, collection=, applications=)`.
- Produces: `KnowledgeAgent._safe_qa_search(self, ctx, query, applications, cfg) -> dict`; `run` performs the qa search, gating, citation merge, and passes `qa_passages`/`qa_leads` to `_compose`.

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_knowledge_qa_merge.py`:

```python
import pytest
from unittest.mock import AsyncMock
import agent_customer_support.agents.knowledge as kn
from agent_customer_support.agents.knowledge import KnowledgeAgent
from agent_customer_support.agents.context import TurnContext
from agent_customer_support.config import get_settings
from agent_customer_support.models import CustomerProfile, SessionState, Conversation

pytestmark = pytest.mark.asyncio


def _ctx(applications=None):
    session = SessionState(conversation_id="c1", selected_applications=applications or [])
    ctx = TurnContext(
        customer=CustomerProfile(customer_id="cust1", name="N"),
        session=session,
        conversation=Conversation(conversation_id="c1", customer_id="cust1"),
        message="làm sao tạo phiếu?",
        transcript="",
    )
    return ctx


def _patch_compose(monkeypatch, agent):
    cap = {}

    async def fake_compose(question, passages, transcript, cfg, allow_clarify=True,
                           qa_passages=None, qa_leads=False):
        cap["passages"] = passages
        cap["qa_passages"] = qa_passages
        cap["qa_leads"] = qa_leads
        return "Anh/Chị vui lòng làm theo hướng dẫn."  # plain answer, no marker

    monkeypatch.setattr(agent, "_compose", fake_compose)
    monkeypatch.setattr(agent, "_contextualize", AsyncMock(return_value="q-standalone"))
    return cap


def _search_dispatch(qa_result, product_result=None):
    cfg = get_settings()
    product_result = product_result or {"passages": ["guide"], "citations": ["g1"], "top_confidence": 0.5}

    async def search(query, collection, applications=None):
        if collection == cfg.qa_collection:
            if isinstance(qa_result, Exception):
                raise qa_result
            return qa_result
        return product_result

    return search


async def test_qa_leads_when_above_threshold(monkeypatch):
    agent = KnowledgeAgent()
    cap = _patch_compose(monkeypatch, agent)
    ctx = _ctx()
    ctx.rag = type("R", (), {})()
    ctx.rag.search = AsyncMock(side_effect=_search_dispatch(
        {"passages": ["cs answer"], "citations": ["abc"], "top_confidence": 0.95}
    ))
    res = await agent.run(ctx)
    assert cap["qa_passages"] == ["cs answer"]
    assert cap["qa_leads"] is True
    assert "qa:abc" in res.citations
    assert "g1" in res.citations


async def test_qa_supplementary_when_below_threshold(monkeypatch):
    agent = KnowledgeAgent()
    cap = _patch_compose(monkeypatch, agent)
    ctx = _ctx()
    ctx.rag = type("R", (), {})()
    ctx.rag.search = AsyncMock(side_effect=_search_dispatch(
        {"passages": ["cs answer"], "citations": ["abc"], "top_confidence": 0.4}
    ))
    await agent.run(ctx)
    assert cap["qa_passages"] == ["cs answer"]
    assert cap["qa_leads"] is False


async def test_qa_search_failure_degrades_to_product_only(monkeypatch):
    agent = KnowledgeAgent()
    cap = _patch_compose(monkeypatch, agent)
    ctx = _ctx()
    ctx.rag = type("R", (), {})()
    ctx.rag.search = AsyncMock(side_effect=_search_dispatch(RuntimeError("no collection")))
    res = await agent.run(ctx)  # must not raise
    assert cap["qa_passages"] == []
    assert cap["qa_leads"] is False
    assert cap["passages"] == ["guide"]


async def test_applications_filter_passed_to_qa_search(monkeypatch):
    agent = KnowledgeAgent()
    _patch_compose(monkeypatch, agent)
    ctx = _ctx(applications=["Lab"])
    ctx.rag = type("R", (), {})()
    ctx.rag.search = AsyncMock(side_effect=_search_dispatch(
        {"passages": [], "citations": [], "top_confidence": 0.0}
    ))
    await agent.run(ctx)
    cfg = get_settings()
    ctx.rag.search.assert_any_await(
        "q-standalone", collection=cfg.qa_collection, applications=["Lab"]
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/agents/test_knowledge_qa_merge.py -v`
Expected: FAIL (qa search not performed / `qa_leads` not passed / no `qa:` citations).

- [ ] **Step 3: Add a module logger**

In `agent_customer_support/agents/knowledge.py`, add at the top with the other imports:

```python
import logging
```

and after the imports / regex definitions (module level, before the class), add:

```python
logger = logging.getLogger(__name__)
```

(If a `logger` already exists, skip this step.)

- [ ] **Step 4: Add `_safe_qa_search` and wire the merge into `run`**

Add this method to `KnowledgeAgent` (e.g. just above `run`):

```python
    async def _safe_qa_search(
        self, ctx: TurnContext, query: str, applications: list[str] | None, cfg: Settings
    ) -> dict:
        """Search the curated Q&A collection, degrading to an empty result on any
        error. The qa collection does not exist until the first CS approval, so a
        missing collection (or any Qdrant error) must never break the guide path."""
        try:
            return await ctx.rag.search(
                query, collection=cfg.qa_collection, applications=applications
            )
        except Exception as exc:  # noqa: BLE001 - degrade, never break the answer
            logger.warning("qa search failed, using product-only: %s", exc)
            return {"passages": [], "citations": [], "top_confidence": 0.0}
```

In `run`, locate the existing block:

```python
        applications = ctx.session.selected_applications or None
        res = await ctx.rag.search(
            query, collection=cfg.product_collection, applications=applications
        )
        passages = res.get("passages", []) or []
        citations = res.get("citations", []) or []
```

and insert immediately after it:

```python
        qa_res = await self._safe_qa_search(ctx, query, applications, cfg)
        qa_passages = qa_res.get("passages", []) or []
        qa_leads = bool(qa_passages) and qa_res.get("top_confidence", 0.0) >= cfg.qa_lead_threshold
        qa_citations = qa_res.get("citations", []) or []
        if qa_citations:
            citations = citations + [f"qa:{c}" for c in qa_citations]
```

Then update the existing `_compose` call in `run` (the one passing `allow_clarify=not already_clarified`) to forward the qa inputs:

```python
        composed = await self._compose(
            query,
            passages,
            ctx.transcript,
            cfg,
            allow_clarify=not already_clarified,
            qa_passages=qa_passages,
            qa_leads=qa_leads,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `poetry run pytest tests/agents/test_knowledge_qa_merge.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Run the existing knowledge tests (no regression)**

Run: `poetry run pytest tests/agents/test_knowledge.py tests/agents/test_knowledge_qa_capture.py -v`
Expected: PASS — these exercise the existing pipeline. If any test mocks `ctx.rag.search` as an `AsyncMock` returning a single dict, it still works because `_safe_qa_search` calls the same mock (now twice, once per collection) and the default return value applies to both. If a test asserts `ctx.rag.search.assert_awaited_once_with(...)`, update that assertion to `assert_any_await(...)` since the qa search adds a second call.

- [ ] **Step 7: Commit**

```bash
git add agent_customer_support/agents/knowledge.py tests/agents/test_knowledge_qa_merge.py
git commit -m "feat(qa): merge qa collection into knowledge retrieval (gated precedence)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Second qa retrieval reusing `RagClient.search` → Task 3. ✓
- Confidence gating (`top_confidence >= qa_lead_threshold`, default 0.85) → Task 1 (config) + Task 3 (gate). ✓
- Three-tier precedence prompt (added top tier, no contradiction) → Task 1 (`KNOWLEDGE_COMPOSE_PROMPT_WITH_QA`). ✓
- Conditional prompt: two-source byte-identical when no qa, three-source when present → Task 2. ✓
- Authoritative vs supplementary CS block (`ưu tiên cao nhất` / `bổ trợ`) → Task 2, keyed to the prompt tokens from Task 1. ✓
- Graceful degradation on qa search failure → Task 3 (`_safe_qa_search`). ✓
- `applications` filter on qa search → Task 3. ✓
- `qa:`-prefixed citations merged → Task 3. ✓
- Untouched: clarify loop / suspected_bug / miss-capture → unchanged in `run` (only the search/compose lines change). ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases"; every code step has complete code. The prompt variant is built via explicit `.replace()` pairs (complete, no transcription of the 40-line prompt). ✓

**Type consistency:** `_compose(..., qa_passages: list[str] | None, qa_leads: bool)` defined in Task 2 matches the call in Task 3. `_safe_qa_search(self, ctx, query, applications, cfg) -> dict` defined and called in Task 3. `KNOWLEDGE_COMPOSE_PROMPT_WITH_QA` produced in Task 1, imported in Task 2. `qa_lead_threshold` produced in Task 1, read in Task 3. Header tokens (`ưu tiên cao nhất`/`bổ trợ`) in Task 2 match the prompt tokens asserted in Task 1. ✓

**Note for executor:** Tests are mock-based (no DynamoDB/Qdrant/LLM). Task 3 Step 6 is the regression guard for the existing pipeline — the only likely edit is upgrading an `assert_awaited_once*` on `ctx.rag.search` to `assert_any_await` because retrieval now calls search twice (product + qa).

# Multi-Agent Re-Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the monolithic `AgentCore` with a deterministic coordinator orchestrating six single-responsibility agents, and vendor a self-contained Anthropic+OpenAI LLM client (with image support) to drop the `enterprise_llm_service` runtime import.

**Architecture:** A no-LLM `Coordinator` loads turn context and routes through a sequential hand-off: `GuardrailAgent` (input) → `TriageAgent` (clarify|route) → `KnowledgeAgent`|`FlowAgent`|`EscalationAgent` → `IssueVerificationAgent` (suspected bugs) → `GuardrailAgent` (output). Agents never call each other; they return a typed `AgentResult` the coordinator branches on. RAG stays behind the existing `RagClient` HTTP interface (Spec B migrates it in-repo later).

**Tech Stack:** Python 3.13, FastAPI, Pydantic v2, pytest + pytest-asyncio + unittest.mock, Anthropic SDK, OpenAI SDK.

**Spec:** `docs/superpowers/specs/2026-06-12-multi-agent-rearchitecture-design.md`

---

## File Structure

```
agent_customer_support/
  llm/                          # NEW vendored client (replaces enterprise_llm_service import)
    __init__.py                 # complete_with_tools(), complete_text()
    normalize.py                # build provider messages; parse provider responses
    providers/
      __init__.py
      anthropic_provider.py     # Anthropic SDK calls
      openai_provider.py        # OpenAI SDK calls
  agents/                       # NEW agent package (replaces agent/core.py)
    __init__.py
    context.py                  # TurnContext dataclass
    base.py                     # Agent protocol
    coordinator.py              # orchestration, no LLM
    guardrail.py
    triage.py
    knowledge.py
    flow.py
    verification.py
    escalation.py
    prompts.py                  # per-agent system prompts
  models.py                     # + Attachment, AgentResult; extend SessionState, ChatRequest, Turn
  llm.py                        # re-point to agent_customer_support.llm
  channels/widget.py            # accept attachments; call Coordinator
  agent/core.py                 # DELETED at the end
```

**Note on TDD:** every agent is tested with a mocked LLM (`complete_with_tools`/`complete_text`) and mocked stores — no network. Follow the existing pattern in `tests/agent/test_core.py` (patch `complete_with_tools`, inject `AsyncMock` stores).

---

## Phase 0 — Models & Contracts

### Task 1: Extend models (Attachment, AgentResult, SessionState, ChatRequest, Turn)

**Files:**
- Modify: `agent_customer_support/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_models.py`:

```python
from agent_customer_support.models import (
    Attachment, AgentResult, SessionState, ChatRequest, Turn,
)


def test_attachment_image():
    a = Attachment(kind="image", media_type="image/png", data="aGVsbG8=")
    assert a.kind == "image"
    assert a.media_type == "image/png"


def test_session_pending_defaults_none():
    s = SessionState(conversation_id="cv1")
    assert s.pending is None
    assert s.pending_context is None


def test_session_pending_verify():
    s = SessionState(conversation_id="cv1", pending="verify_issue",
                     pending_context={"summary": "x", "module": "m"})
    assert s.pending == "verify_issue"
    assert s.pending_context["summary"] == "x"


def test_chat_request_attachments_default_empty():
    r = ChatRequest(customer_id="c1", conversation_id="cv1", message="hi")
    assert r.attachments == []


def test_agent_result_defaults():
    r = AgentResult(reply="hello")
    assert r.action == "reply"
    assert r.routed_to is None
    assert r.suspected_bug is False
    assert r.evidence_complete is False
    assert r.escalated is False


def test_agent_result_route():
    r = AgentResult(action="route", routed_to="knowledge")
    assert r.action == "route"
    assert r.routed_to == "knowledge"


def test_turn_attachments_default_empty():
    t = Turn(role="user", content="hi")
    assert t.attachments == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_models.py -v`
Expected: FAIL — `ImportError: cannot import name 'Attachment'` / `AgentResult`.

- [ ] **Step 3: Write minimal implementation**

In `agent_customer_support/models.py`, add after the existing imports/`_now`:

```python
# ---- Attachments ----


class Attachment(BaseModel):
    kind: Literal["image"]
    media_type: str          # image/png | image/jpeg
    data: str                # base64-encoded bytes
```

Modify `Turn` to carry attachments:

```python
class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    attachments: list[Attachment] = Field(default_factory=list)
    ts: datetime = Field(default_factory=_now)
```

Modify `SessionState`:

```python
class SessionState(BaseModel):
    conversation_id: str
    active_flow_id: str | None = None
    current_step_id: str | None = None
    pending: Literal["verify_issue"] | None = None
    pending_context: dict | None = None
    updated_at: datetime = Field(default_factory=_now)
```

Modify `ChatRequest`:

```python
class ChatRequest(BaseModel):
    customer_id: str
    conversation_id: str
    message: str
    attachments: list[Attachment] = Field(default_factory=list)
```

Add at the end of the file:

```python
# ---- Agent contract ----


class AgentResult(BaseModel):
    action: Literal["reply", "route"] = "reply"
    reply: str = ""
    routed_to: Literal["knowledge", "flow", "escalate"] | None = None
    resolved: bool | None = None
    suspected_bug: bool = False
    evidence_complete: bool = False
    evidence: dict | None = None
    escalated: bool = False
    new_session: SessionState | None = None
    citations: list[str] = Field(default_factory=list)
```

(`Attachment` must be defined before `Turn`; place the Attachments section above the Conversation section.)

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_models.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent_customer_support/models.py tests/test_models.py
git commit -m "feat: add Attachment, AgentResult and multi-turn session state"
```

---

### Task 2: TurnContext + Agent protocol

**Files:**
- Create: `agent_customer_support/agents/__init__.py`
- Create: `agent_customer_support/agents/context.py`
- Create: `agent_customer_support/agents/base.py`
- Test: `tests/agents/__init__.py`, `tests/agents/test_base.py`

- [ ] **Step 1: Write the failing test**

Create `tests/agents/__init__.py` (empty) and `tests/agents/test_base.py`:

```python
from unittest.mock import AsyncMock
from agent_customer_support.agents.context import TurnContext
from agent_customer_support.agents.base import Agent
from agent_customer_support.models import (
    AgentResult, CustomerProfile, SessionState, Conversation,
)


def _ctx() -> TurnContext:
    return TurnContext(
        customer=CustomerProfile(customer_id="c1", name="C1", enabled_modules=["m"]),
        session=SessionState(conversation_id="cv1"),
        conversation=Conversation(conversation_id="cv1", customer_id="c1"),
        message="hi",
        attachments=[],
        transcript="user: hi",
        rag=AsyncMock(),
        flow_store=AsyncMock(),
        backlog=AsyncMock(),
        escalator=AsyncMock(),
    )


def test_turn_context_fields():
    ctx = _ctx()
    assert ctx.message == "hi"
    assert ctx.customer.customer_id == "c1"
    assert ctx.session.conversation_id == "cv1"


def test_agent_protocol_is_runtime_checkable():
    class Dummy:
        name = "dummy"

        async def run(self, ctx: TurnContext) -> AgentResult:
            return AgentResult(reply="ok")

    assert isinstance(Dummy(), Agent)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/agents/test_base.py -v`
Expected: FAIL — module `agent_customer_support.agents.context` not found.

- [ ] **Step 3: Write minimal implementation**

Create `agent_customer_support/agents/__init__.py` (empty).

Create `agent_customer_support/agents/context.py`:

```python
from dataclasses import dataclass, field
from typing import Any

from agent_customer_support.models import (
    Attachment, Conversation, CustomerProfile, SessionState,
)


@dataclass
class TurnContext:
    customer: CustomerProfile
    session: SessionState
    conversation: Conversation
    message: str
    attachments: list[Attachment] = field(default_factory=list)
    transcript: str = ""
    # shared service handles (typed Any to avoid import cycles with stores)
    rag: Any = None
    flow_store: Any = None
    backlog: Any = None
    escalator: Any = None
```

Create `agent_customer_support/agents/base.py`:

```python
from typing import Protocol, runtime_checkable

from agent_customer_support.agents.context import TurnContext
from agent_customer_support.models import AgentResult


@runtime_checkable
class Agent(Protocol):
    name: str

    async def run(self, ctx: TurnContext) -> AgentResult: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/agents/test_base.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent_customer_support/agents/ tests/agents/
git commit -m "feat: add TurnContext and Agent protocol"
```

---

## Phase 1 — Vendored LLM Client

The client exposes the **same contract** `core.py` used: `complete_with_tools(messages, tools, system) -> {stop_reason, text, tool_calls}` and `complete_text(messages) -> str`. Provider is chosen from `settings.agent_model` (`claude*` → Anthropic, else OpenAI).

### Task 3: Message normalization (text + image blocks)

**Files:**
- Create: `agent_customer_support/llm/__init__.py` (empty for now)
- Create: `agent_customer_support/llm/providers/__init__.py` (empty)
- Create: `agent_customer_support/llm/normalize.py`
- Test: `tests/llm/__init__.py`, `tests/llm/test_normalize.py`

- [ ] **Step 1: Write the failing test**

Create `tests/llm/__init__.py` (empty) and `tests/llm/test_normalize.py`:

```python
from agent_customer_support.llm.normalize import (
    to_anthropic_content, to_openai_content,
)
from agent_customer_support.models import Attachment


def test_anthropic_text_only():
    blocks = to_anthropic_content("hello", [])
    assert blocks == [{"type": "text", "text": "hello"}]


def test_anthropic_with_image():
    att = Attachment(kind="image", media_type="image/png", data="QUJD")
    blocks = to_anthropic_content("see this", [att])
    assert blocks[0] == {"type": "text", "text": "see this"}
    assert blocks[1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"},
    }


def test_openai_text_only():
    content = to_openai_content("hello", [])
    assert content == "hello"


def test_openai_with_image():
    att = Attachment(kind="image", media_type="image/jpeg", data="QUJD")
    content = to_openai_content("see this", [att])
    assert content[0] == {"type": "text", "text": "see this"}
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,QUJD"},
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/llm/test_normalize.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

Create `agent_customer_support/llm/__init__.py` (empty), `agent_customer_support/llm/providers/__init__.py` (empty).

Create `agent_customer_support/llm/normalize.py`:

```python
from agent_customer_support.models import Attachment


def to_anthropic_content(text: str, attachments: list[Attachment]) -> list[dict]:
    blocks: list[dict] = [{"type": "text", "text": text}]
    for a in attachments:
        if a.kind == "image":
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": a.media_type,
                    "data": a.data,
                },
            })
    return blocks


def to_openai_content(text: str, attachments: list[Attachment]):
    if not attachments:
        return text
    content: list[dict] = [{"type": "text", "text": text}]
    for a in attachments:
        if a.kind == "image":
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{a.media_type};base64,{a.data}"},
            })
    return content
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/llm/test_normalize.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent_customer_support/llm/ tests/llm/
git commit -m "feat: add LLM message normalization with image support"
```

---

### Task 4: Anthropic provider (tool-use + response parse)

**Files:**
- Create: `agent_customer_support/llm/providers/anthropic_provider.py`
- Test: `tests/llm/test_anthropic_provider.py`

The provider returns `{"stop_reason": str, "text": str | None, "tool_calls": [{"id","name","input"}]}`. The Anthropic SDK client is injected so tests use a fake.

- [ ] **Step 1: Write the failing test**

Create `tests/llm/test_anthropic_provider.py`:

```python
from types import SimpleNamespace
from unittest.mock import MagicMock
from agent_customer_support.llm.providers.anthropic_provider import (
    anthropic_complete_with_tools,
)


def _block(**kw):
    return SimpleNamespace(**kw)


def test_parses_text_response():
    resp = SimpleNamespace(
        stop_reason="end_turn",
        content=[_block(type="text", text="hello")],
    )
    client = MagicMock()
    client.messages.create.return_value = resp
    out = anthropic_complete_with_tools(
        client=client, model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "hi"}],
        tools=[], system="sys",
    )
    assert out["stop_reason"] == "end_turn"
    assert out["text"] == "hello"
    assert out["tool_calls"] == []


def test_parses_tool_use():
    resp = SimpleNamespace(
        stop_reason="tool_use",
        content=[
            _block(type="text", text="let me check"),
            _block(type="tool_use", id="t1", name="search_knowledge",
                   input={"query": "x"}),
        ],
    )
    client = MagicMock()
    client.messages.create.return_value = resp
    out = anthropic_complete_with_tools(
        client=client, model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"name": "search_knowledge"}], system=None,
    )
    assert out["stop_reason"] == "tool_use"
    assert out["text"] == "let me check"
    assert out["tool_calls"] == [
        {"id": "t1", "name": "search_knowledge", "input": {"query": "x"}}
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/llm/test_anthropic_provider.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

Create `agent_customer_support/llm/providers/anthropic_provider.py`:

```python
def anthropic_complete_with_tools(
    *, client, model: str, messages: list[dict],
    tools: list[dict], system: str | None, max_tokens: int = 1500,
) -> dict:
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if tools:
        kwargs["tools"] = tools
    if system:
        kwargs["system"] = system

    resp = client.messages.create(**kwargs)

    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in resp.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append({"id": block.id, "name": block.name, "input": block.input})

    return {
        "stop_reason": resp.stop_reason,
        "text": "".join(text_parts) or None,
        "tool_calls": tool_calls,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/llm/test_anthropic_provider.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent_customer_support/llm/providers/anthropic_provider.py tests/llm/test_anthropic_provider.py
git commit -m "feat: add Anthropic provider with tool-use parsing"
```

---

### Task 5: OpenAI provider (tool_calls + response parse)

**Files:**
- Create: `agent_customer_support/llm/providers/openai_provider.py`
- Test: `tests/llm/test_openai_provider.py`

OpenAI tool schema differs (wrapped in `{"type":"function","function":{...}}`); the provider converts the Anthropic-style `TOOL_DEFS` (`{name, description, input_schema}`) to OpenAI's shape, then normalizes the response back to `{stop_reason, text, tool_calls}` where `stop_reason="tool_use"` when tool calls are present.

- [ ] **Step 1: Write the failing test**

Create `tests/llm/test_openai_provider.py`:

```python
import json
from types import SimpleNamespace
from unittest.mock import MagicMock
from agent_customer_support.llm.providers.openai_provider import (
    openai_complete_with_tools, to_openai_tools,
)


def test_to_openai_tools_shape():
    defs = [{"name": "f", "description": "d", "input_schema": {"type": "object"}}]
    out = to_openai_tools(defs)
    assert out == [{
        "type": "function",
        "function": {"name": "f", "description": "d", "parameters": {"type": "object"}},
    }]


def test_parses_text_response():
    msg = SimpleNamespace(content="hello", tool_calls=None)
    resp = SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])
    client = MagicMock()
    client.chat.completions.create.return_value = resp
    out = openai_complete_with_tools(
        client=client, model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}], tools=[], system="sys",
    )
    assert out["stop_reason"] == "stop"
    assert out["text"] == "hello"
    assert out["tool_calls"] == []


def test_parses_tool_calls():
    tc = SimpleNamespace(
        id="t1",
        function=SimpleNamespace(name="search_knowledge",
                                 arguments=json.dumps({"query": "x"})),
    )
    msg = SimpleNamespace(content=None, tool_calls=[tc])
    resp = SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")])
    client = MagicMock()
    client.chat.completions.create.return_value = resp
    out = openai_complete_with_tools(
        client=client, model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"name": "search_knowledge", "description": "d", "input_schema": {}}],
        system=None,
    )
    assert out["stop_reason"] == "tool_use"
    assert out["tool_calls"] == [
        {"id": "t1", "name": "search_knowledge", "input": {"query": "x"}}
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/llm/test_openai_provider.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

Create `agent_customer_support/llm/providers/openai_provider.py`:

```python
import json


def to_openai_tools(tool_defs: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object"}),
            },
        }
        for t in tool_defs
    ]


def openai_complete_with_tools(
    *, client, model: str, messages: list[dict],
    tools: list[dict], system: str | None, max_tokens: int = 1500,
) -> dict:
    msgs = list(messages)
    if system:
        msgs = [{"role": "system", "content": system}, *msgs]

    kwargs: dict = {"model": model, "messages": msgs, "max_tokens": max_tokens}
    if tools:
        kwargs["tools"] = to_openai_tools(tools)

    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    msg = choice.message

    tool_calls: list[dict] = []
    for tc in (msg.tool_calls or []):
        tool_calls.append({
            "id": tc.id,
            "name": tc.function.name,
            "input": json.loads(tc.function.arguments or "{}"),
        })

    stop_reason = "tool_use" if tool_calls else choice.finish_reason
    return {
        "stop_reason": stop_reason,
        "text": msg.content,
        "tool_calls": tool_calls,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/llm/test_openai_provider.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent_customer_support/llm/providers/openai_provider.py tests/llm/test_openai_provider.py
git commit -m "feat: add OpenAI provider with tool_calls normalization"
```

---

### Task 6: LLM facade + swap `llm.py`

**Files:**
- Modify: `agent_customer_support/llm/__init__.py`
- Modify: `agent_customer_support/llm.py`
- Test: `tests/llm/test_facade.py`, `tests/test_llm.py` (update)

**Important:** there is both a module `agent_customer_support/llm.py` and a package `agent_customer_support/llm/`. Python cannot have both. **Delete `agent_customer_support/llm.py`** and move its two public functions into `agent_customer_support/llm/__init__.py`. Update all imports `from agent_customer_support.llm import complete_with_tools, complete_text` — they continue to work because the package `__init__` exports the same names.

- [ ] **Step 1: Write the failing test**

Create `tests/llm/test_facade.py`:

```python
from unittest.mock import patch, MagicMock
from agent_customer_support.llm import complete_with_tools, complete_text


def test_routes_to_anthropic_for_claude_model():
    fake = {"stop_reason": "end_turn", "text": "hi", "tool_calls": []}
    with patch("agent_customer_support.llm.get_settings") as gs, \
         patch("agent_customer_support.llm._anthropic_client", return_value=MagicMock()), \
         patch("agent_customer_support.llm.anthropic_complete_with_tools",
               return_value=fake) as m:
        gs.return_value.agent_model = "claude-3-5-sonnet"
        out = complete_with_tools(messages=[{"role": "user", "content": "x"}],
                                  tools=[], system=None)
    assert out["text"] == "hi"
    m.assert_called_once()


def test_routes_to_openai_for_gpt_model():
    fake = {"stop_reason": "stop", "text": "hi", "tool_calls": []}
    with patch("agent_customer_support.llm.get_settings") as gs, \
         patch("agent_customer_support.llm._openai_client", return_value=MagicMock()), \
         patch("agent_customer_support.llm.openai_complete_with_tools",
               return_value=fake) as m:
        gs.return_value.agent_model = "gpt-4o-mini"
        out = complete_with_tools(messages=[{"role": "user", "content": "x"}],
                                  tools=[], system=None)
    assert out["text"] == "hi"
    m.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/llm/test_facade.py -v`
Expected: FAIL — names `_anthropic_client` etc. not defined.

- [ ] **Step 3: Write minimal implementation**

Delete the old module:

```bash
git rm agent_customer_support/llm.py
```

Write `agent_customer_support/llm/__init__.py`:

```python
from functools import lru_cache

from agent_customer_support.config import get_settings
from agent_customer_support.llm.providers.anthropic_provider import (
    anthropic_complete_with_tools,
)
from agent_customer_support.llm.providers.openai_provider import (
    openai_complete_with_tools,
)


@lru_cache
def _anthropic_client():
    from anthropic import Anthropic
    return Anthropic()  # reads ANTHROPIC_API_KEY from env


@lru_cache
def _openai_client():
    from openai import OpenAI
    return OpenAI()  # reads OPENAI_API_KEY from env


def _is_anthropic(model: str) -> bool:
    return "claude" in model


def complete_with_tools(
    *, messages: list[dict], tools: list[dict], system: str | None = None
) -> dict:
    model = get_settings().agent_model
    if _is_anthropic(model):
        return anthropic_complete_with_tools(
            client=_anthropic_client(), model=model,
            messages=messages, tools=tools, system=system,
        )
    return openai_complete_with_tools(
        client=_openai_client(), model=model,
        messages=messages, tools=tools, system=system,
    )


def complete_text(messages: list[dict], system: str | None = None) -> str:
    out = complete_with_tools(messages=messages, tools=[], system=system)
    return out.get("text") or ""
```

- [ ] **Step 4: Run tests**

Run: `poetry run pytest tests/llm/ -v`
Expected: PASS.

Update `tests/test_llm.py` if it patches the old `enterprise_llm_service` symbols — re-point any patch targets to `agent_customer_support.llm.anthropic_complete_with_tools` / `openai_complete_with_tools`. Run:

Run: `poetry run pytest tests/test_llm.py -v`
Expected: PASS (or delete obsolete assertions tied to the old wrapper).

- [ ] **Step 5: Add SDK deps + commit**

```bash
poetry add anthropic openai
git add agent_customer_support/llm/ tests/llm/ tests/test_llm.py pyproject.toml poetry.lock
git commit -m "feat: vendor LLM client, drop enterprise_llm_service runtime import"
```

---

## Phase 2 — Agents (one responsibility each)

All agents share the `complete_with_tools`/`complete_text` facade and the existing `TOOL_DEFS`/`dispatch` from `agent/tools.py`. The per-agent prompts live in `agents/prompts.py`.

### Task 7: Per-agent prompts module

**Files:**
- Create: `agent_customer_support/agents/prompts.py`
- Test: `tests/agents/test_prompts.py`

This splits the single `_BASE` in `agent/prompt.py` into focused prompts. Keep `agent/prompt.py` for the flow-context helper (`build_system_prompt`) used by `FlowAgent`.

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_prompts.py`:

```python
from agent_customer_support.agents.prompts import (
    TRIAGE_PROMPT, VERIFICATION_PROMPT, GUARDRAIL_OUTPUT_PROMPT,
    KNOWLEDGE_GRADER_PROMPT, KNOWLEDGE_REFORMULATE_PROMPT, KNOWLEDGE_COMPOSE_PROMPT,
)


def test_prompts_are_nonempty_strings():
    for p in (TRIAGE_PROMPT, VERIFICATION_PROMPT, GUARDRAIL_OUTPUT_PROMPT,
              KNOWLEDGE_GRADER_PROMPT, KNOWLEDGE_REFORMULATE_PROMPT,
              KNOWLEDGE_COMPOSE_PROMPT):
        assert isinstance(p, str) and len(p) > 20


def test_triage_mentions_clarify_and_route():
    assert "clarify" in TRIAGE_PROMPT.lower()
    assert "route" in TRIAGE_PROMPT.lower()


def test_grader_judges_content_not_score():
    assert "answer_present" in KNOWLEDGE_GRADER_PROMPT


def test_compose_has_no_answer_and_bug_markers():
    assert "[[no_answer]]" in KNOWLEDGE_COMPOSE_PROMPT
    assert "suspected_bug" in KNOWLEDGE_COMPOSE_PROMPT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/agents/test_prompts.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

Create `agent_customer_support/agents/prompts.py`:

```python
TRIAGE_PROMPT = """Bạn là bộ định tuyến (triage) cho trợ lý hỗ trợ phần mềm CenLab.
Nhiệm vụ DUY NHẤT: quyết định nên LÀM RÕ (clarify) hay ĐỊNH TUYẾN (route).

Phần lớn câu hỏi đầu tiên của người dùng KHÔNG rõ ràng. Khi mục tiêu của người dùng
chưa rõ → trả về action "clarify" kèm MỘT câu hỏi ngắn để hiểu ý định.

Khi đã rõ ý định → action "route" với target:
- "knowledge": mọi câu hỏi nghiệp vụ, "cách làm", báo lỗi, đề xuất tính năng.
  LƯU Ý: lời than phiền ("bị lỗi", "không chạy được", "thêm tính năng") KHÔNG được
  route thẳng tới escalate — luôn để knowledge thử giải quyết trước.
- "escalate": CHỈ khi người dùng nói rõ muốn gặp nhân viên/người thật.

Trả về JSON: {"action":"clarify","question":"..."} hoặc {"action":"route","target":"knowledge|escalate"}.
"""

# KnowledgeAgent is an orchestrated pipeline making three distinct LLM calls:
# GRADER (answer-presence), REFORMULATE (jargon -> product terms), COMPOSE (grounded answer).

KNOWLEDGE_GRADER_PROMPT = """Bạn là bộ chấm điểm độ liên quan cho RAG của phần mềm CenLab.
Cho CÂU HỎI và các ĐOẠN TRÍCH (passages), hãy quyết định: các đoạn này có chứa câu trả lời
TRỰC TIẾP cho câu hỏi không? Điểm tương đồng (similarity) KHÔNG quan trọng — chỉ xét NỘI DUNG.
Trả về JSON: {"answer_present": true|false, "reason": "..."}.
"""

KNOWLEDGE_REFORMULATE_PROMPT = """Người dùng thường dùng thuật ngữ riêng của công ty họ, không khớp
từ ngữ trong tài liệu phần mềm CenLab. Viết lại câu hỏi sang từ ngữ/khái niệm của phần mềm CenLab
để tìm kiếm tốt hơn. Dùng danh sách module đang bật và ghi chú cấu hình (nếu có) làm gợi ý ánh xạ.
Chỉ trả về MỘT câu truy vấn đã viết lại, không giải thích.
"""

KNOWLEDGE_COMPOSE_PROMPT = """Bạn là trợ lý hỗ trợ phần mềm CenLab của Tâm Đức.
Trả lời bằng tiếng Việt, ngắn gọn, CHỈ dựa trên các đoạn trích được cung cấp.

- Nếu các đoạn trích KHÔNG thực sự trả lời câu hỏi → KẾT THÚC bằng marker [[no_answer]] (đừng bịa).
- Nếu tài liệu xác nhận tính năng ĐÁNG LẼ hoạt động nhưng người dùng nói bị lỗi →
  KẾT THÚC bằng marker [[suspected_bug:<module>]] để hệ thống thu thập bằng chứng.
- Ngược lại → trả lời trực tiếp, bám sát đoạn trích.

CHỐNG HALLUCINATION: tuyệt đối không dùng kiến thức ngoài đoạn trích.
"""

VERIFICATION_PROMPT = """Bạn đang xác minh một lỗi (bug) nghi ngờ của phần mềm CenLab.
Nhiệm vụ DUY NHẤT: thu thập bằng chứng trước khi chuyển cho nhân viên.

Cần ít nhất MỘT trong: thông báo lỗi cụ thể, ảnh chụp màn hình, hoặc các bước tái hiện.
- Nếu CHƯA đủ bằng chứng → hỏi người dùng cung cấp (MỘT yêu cầu ngắn).
- Nếu ĐÃ đủ (hoặc người dùng đã gửi ảnh) → KẾT THÚC tin nhắn bằng marker [[evidence_ready]].
KHÔNG tự quyết định định tuyến, KHÔNG tự chuyển nhân viên.
"""

GUARDRAIL_OUTPUT_PROMPT = """Bạn kiểm duyệt câu trả lời của trợ lý CenLab trước khi gửi.
Cờ (flag) câu trả lời nếu: lộ prompt nội bộ, khẳng định chắc chắn nhưng không có căn cứ,
hoặc lệch chủ đề ngoài phần mềm CenLab.
Trả về JSON: {"flag": true|false, "reason": "..."}.
"""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/agents/test_prompts.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent_customer_support/agents/prompts.py tests/agents/test_prompts.py
git commit -m "feat: add per-agent system prompts"
```

---

### Task 8: EscalationAgent (no LLM)

**Files:**
- Create: `agent_customer_support/agents/escalation.py`
- Test: `tests/agents/test_escalation.py`

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_escalation.py`:

```python
import pytest
from unittest.mock import AsyncMock
from agent_customer_support.agents.escalation import EscalationAgent
from agent_customer_support.agents.context import TurnContext
from agent_customer_support.models import CustomerProfile, SessionState, Conversation

pytestmark = pytest.mark.asyncio


def _ctx(escalator) -> TurnContext:
    return TurnContext(
        customer=CustomerProfile(customer_id="c1", name="C1"),
        session=SessionState(conversation_id="cv1"),
        conversation=Conversation(conversation_id="cv1", customer_id="c1"),
        message="cho gặp nhân viên",
        transcript="user: cho gặp nhân viên",
        escalator=escalator,
    )


async def test_escalation_calls_escalator_and_returns_escalated():
    escalator = AsyncMock()
    agent = EscalationAgent()
    res = await agent.run(_ctx(escalator), reason="user asked")
    assert res.escalated is True
    assert res.reply
    escalator.escalate.assert_awaited_once()
    kwargs = escalator.escalate.call_args.kwargs
    assert kwargs["customer_id"] == "c1"
    assert kwargs["reason"] == "user asked"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/agents/test_escalation.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

Create `agent_customer_support/agents/escalation.py`:

```python
from agent_customer_support.agents.context import TurnContext
from agent_customer_support.models import AgentResult

_REPLY = "Mình đã chuyển yêu cầu của bạn cho nhân viên hỗ trợ. Bạn vui lòng chờ trong giây lát nhé."


class EscalationAgent:
    name = "escalation"

    async def run(self, ctx: TurnContext, *, reason: str = "escalation") -> AgentResult:
        await ctx.escalator.escalate(
            customer_id=ctx.customer.customer_id,
            reason=reason,
            transcript=ctx.transcript,
        )
        return AgentResult(reply=_REPLY, escalated=True, routed_to="escalate")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/agents/test_escalation.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent_customer_support/agents/escalation.py tests/agents/test_escalation.py
git commit -m "feat: add EscalationAgent"
```

---

### Task 9: GuardrailAgent (rules-first input + LLM output check)

**Files:**
- Create: `agent_customer_support/agents/guardrail.py`
- Test: `tests/agents/test_guardrail.py`

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_guardrail.py`:

```python
import pytest
from unittest.mock import patch
from agent_customer_support.agents.guardrail import GuardrailAgent

pytestmark = pytest.mark.asyncio


async def test_empty_input_blocked():
    g = GuardrailAgent()
    res = await g.check_input("   ")
    assert res["pass"] is False


async def test_oversized_input_blocked():
    g = GuardrailAgent()
    res = await g.check_input("x" * 6000)
    assert res["pass"] is False


async def test_normal_input_passes():
    g = GuardrailAgent()
    res = await g.check_input("làm sao tạo phiếu yêu cầu?")
    assert res["pass"] is True


async def test_output_flagged_by_llm():
    g = GuardrailAgent()
    with patch("agent_customer_support.agents.guardrail.complete_text",
               return_value='{"flag": true, "reason": "off-topic"}'):
        res = await g.check_output("nội dung ngoài phạm vi")
    assert res["pass"] is False
    assert res["reason"] == "off-topic"


async def test_output_passes_when_not_flagged():
    g = GuardrailAgent()
    with patch("agent_customer_support.agents.guardrail.complete_text",
               return_value='{"flag": false, "reason": ""}'):
        res = await g.check_output("câu trả lời hợp lệ")
    assert res["pass"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/agents/test_guardrail.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

Create `agent_customer_support/agents/guardrail.py`:

```python
import json

from agent_customer_support.agents.prompts import GUARDRAIL_OUTPUT_PROMPT
from agent_customer_support.llm import complete_text

MAX_INPUT_CHARS = 5000


class GuardrailAgent:
    name = "guardrail"

    async def check_input(self, message: str) -> dict:
        text = (message or "").strip()
        if not text:
            return {"pass": False, "reason": "empty_input"}
        if len(text) > MAX_INPUT_CHARS:
            return {"pass": False, "reason": "oversized_input"}
        return {"pass": True, "reason": ""}

    async def check_output(self, reply: str) -> dict:
        raw = complete_text(
            messages=[{"role": "user", "content": reply}],
            system=GUARDRAIL_OUTPUT_PROMPT,
        )
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {"pass": True, "reason": ""}  # fail-open on parse error
        if data.get("flag"):
            return {"pass": False, "reason": data.get("reason", "flagged")}
        return {"pass": True, "reason": ""}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/agents/test_guardrail.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent_customer_support/agents/guardrail.py tests/agents/test_guardrail.py
git commit -m "feat: add GuardrailAgent (rules input + LLM output check)"
```

---

### Task 10: TriageAgent (rules fast-path + LLM clarify/route)

**Files:**
- Create: `agent_customer_support/agents/triage.py`
- Test: `tests/agents/test_triage.py`

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_triage.py`:

```python
import pytest
from unittest.mock import patch, AsyncMock
from agent_customer_support.agents.triage import TriageAgent
from agent_customer_support.agents.context import TurnContext
from agent_customer_support.models import CustomerProfile, SessionState, Conversation

pytestmark = pytest.mark.asyncio


def _ctx(message, session=None) -> TurnContext:
    return TurnContext(
        customer=CustomerProfile(customer_id="c1", name="C1"),
        session=session or SessionState(conversation_id="cv1"),
        conversation=Conversation(conversation_id="cv1", customer_id="c1"),
        message=message,
        transcript=f"user: {message}",
        rag=AsyncMock(), flow_store=AsyncMock(),
    )


async def test_active_flow_routes_to_flow_without_llm():
    s = SessionState(conversation_id="cv1", active_flow_id="f1", current_step_id="s1")
    res = await TriageAgent().run(_ctx("ok rồi", s))
    assert res.action == "route" and res.routed_to == "flow"


async def test_explicit_human_request_routes_escalate():
    res = await TriageAgent().run(_ctx("cho tôi gặp nhân viên"))
    assert res.action == "route" and res.routed_to == "escalate"


async def test_ambiguous_message_clarifies():
    with patch("agent_customer_support.agents.triage.complete_text",
               return_value='{"action":"clarify","question":"Bạn muốn làm gì cụ thể?"}'):
        res = await TriageAgent().run(_ctx("phần mềm có vấn đề"))
    assert res.action == "reply"
    assert "cụ thể" in res.reply


async def test_clear_intent_routes_knowledge():
    with patch("agent_customer_support.agents.triage.complete_text",
               return_value='{"action":"route","target":"knowledge"}'):
        res = await TriageAgent().run(_ctx("cách tạo phiếu yêu cầu thử nghiệm?"))
    assert res.action == "route" and res.routed_to == "knowledge"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/agents/test_triage.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

Create `agent_customer_support/agents/triage.py`:

```python
import json
import re

from agent_customer_support.agents.context import TurnContext
from agent_customer_support.agents.prompts import TRIAGE_PROMPT
from agent_customer_support.llm import complete_text
from agent_customer_support.models import AgentResult

_HUMAN_RE = re.compile(r"(gặp|cho).{0,12}(nhân viên|người thật|tư vấn viên|cs)", re.I)


class TriageAgent:
    name = "triage"

    async def run(self, ctx: TurnContext) -> AgentResult:
        # Rule fast-paths (no LLM)
        if ctx.session.active_flow_id:
            return AgentResult(action="route", routed_to="flow")
        if _HUMAN_RE.search(ctx.message or ""):
            return AgentResult(action="route", routed_to="escalate")

        raw = complete_text(
            messages=[{"role": "user", "content": ctx.message}],
            system=TRIAGE_PROMPT,
        )
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            data = {"action": "route", "target": "knowledge"}  # fail-safe default

        if data.get("action") == "clarify":
            return AgentResult(action="reply", reply=data.get("question", ""))

        target = data.get("target", "knowledge")
        if target not in ("knowledge", "flow", "escalate"):
            target = "knowledge"
        return AgentResult(action="route", routed_to=target)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/agents/test_triage.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent_customer_support/agents/triage.py tests/agents/test_triage.py
git commit -m "feat: add TriageAgent (rules fast-path + clarify/route)"
```

---

### Task 11a: Reframe `grounding_note` as a neutral hint

**Files:**
- Modify: `agent_customer_support/rag_client.py:42-63`
- Test: `tests/test_rag_client.py`

The score-threshold instructions baked into `grounding_note` ("ask clarification" / "call log_request") move OUT of `rag_client` — that decision now belongs to `KnowledgeAgent`. `rag_client` returns the score as a neutral hint only.

- [ ] **Step 1: Update the test**

In `tests/test_rag_client.py`, replace assertions that check for "clarification"/"log_request" wording in `grounding_note` with:

```python
def test_grounding_note_is_neutral_hint():
    # after building a result with top_confidence ~0.7
    # grounding_note must NOT prescribe an action; it states the score + defers judgment
    note = res["grounding_note"]
    assert "log_request" not in note
    assert "clarification" not in note
    assert "0.7" in note or "confidence" in note.lower()
```

(Keep the rest of the existing rag_client test setup; only the grounding_note expectations change.)

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_rag_client.py -v`
Expected: FAIL — current note contains "clarification"/"log_request".

- [ ] **Step 3: Replace the grounding_note block**

In `agent_customer_support/rag_client.py`, replace the threshold branches (lines ~46-56) with a single neutral note:

```python
        # grounding_note is a HINT ONLY. The similarity score does not tell you whether
        # the passages actually answer the question — KnowledgeAgent judges that from the
        # passage text. We just surface the score and defer the decision.
        grounding = (
            f"confidence={top_conf:.2f} (chỉ là gợi ý độ tương đồng, KHÔNG phải độ đúng). "
            "Hãy tự đánh giá các passages có TRỰC TIẾP trả lời câu hỏi hay không."
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_rag_client.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent_customer_support/rag_client.py tests/test_rag_client.py
git commit -m "refactor: make grounding_note a neutral score hint"
```

---

### Task 11b: KnowledgeAgent (orchestrated pipeline: hybrid grader + reformulation)

**Files:**
- Create: `agent_customer_support/agents/knowledge.py`
- Test: `tests/agents/test_knowledge.py`

KnowledgeAgent is a **deterministic pipeline** (not a model-driven tool loop), so the relevance gate is explicit and testable. It calls `ctx.rag.search()` directly, decides answer-presence with a hybrid grader, reformulates once on miss, then composes a grounded answer. Three pure-ish helpers are unit-tested without an LLM; the LLM steps (`_grade`, `_reformulate`, `_compose`) are patched individually.

Conflict rule (`needs_grading`): grade only when the score is unreliable — medium band `[0.50, 0.80)`, or high score `>= 0.80` with suspiciously short/few passages.

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_knowledge.py`:

```python
import pytest
from unittest.mock import patch, AsyncMock
from agent_customer_support.agents.knowledge import (
    KnowledgeAgent, parse_markers, needs_grading,
)
from agent_customer_support.agents.context import TurnContext
from agent_customer_support.models import CustomerProfile, SessionState, Conversation

pytestmark = pytest.mark.asyncio


def _ctx(message="cách tạo phiếu?") -> TurnContext:
    return TurnContext(
        customer=CustomerProfile(customer_id="c1", name="C1", enabled_modules=["m"]),
        session=SessionState(conversation_id="cv1"),
        conversation=Conversation(conversation_id="cv1", customer_id="c1"),
        message=message,
        transcript=f"user: {message}",
        rag=AsyncMock(), backlog=AsyncMock(), flow_store=AsyncMock(),
    )


# ---- pure helpers ----

def test_needs_grading_medium_band_true():
    assert needs_grading(0.70, ["a" * 500]) is True


def test_needs_grading_high_long_passages_false():
    assert needs_grading(0.88, ["a" * 500]) is False


def test_needs_grading_high_but_short_passages_true():
    assert needs_grading(0.88, ["short"]) is True


def test_needs_grading_no_passages_false():
    assert needs_grading(0.70, []) is False


def test_needs_grading_low_false():
    assert needs_grading(0.30, ["a" * 500]) is False


def test_parse_markers_no_answer():
    clean, kind, mod = parse_markers("Không rõ. [[no_answer]]")
    assert kind == "no_answer" and "[[no_answer]]" not in clean


def test_parse_markers_suspected_bug():
    clean, kind, mod = parse_markers("Đáng lẽ chạy. [[suspected_bug:xet-nghiem]]")
    assert kind == "suspected_bug" and mod == "xet-nghiem"


def test_parse_markers_plain_answer():
    clean, kind, mod = parse_markers("Vào menu X.")
    assert kind is None and mod is None and clean == "Vào menu X."


# ---- pipeline branches ----

async def test_high_confidence_composes_answer():
    ctx = _ctx()
    ctx.rag.search.return_value = {
        "passages": ["x" * 500], "citations": ["c#1"], "top_confidence": 0.9,
    }
    with patch("agent_customer_support.agents.knowledge.complete_text",
               return_value="Vào menu X rồi tạo."):
        res = await KnowledgeAgent().run(ctx)
    assert res.resolved is True
    assert "menu X" in res.reply
    ctx.rag.search.assert_awaited_once()  # no reformulation needed


async def test_medium_band_grader_present_then_answer():
    ctx = _ctx("thuật ngữ riêng của cty")
    ctx.rag.search.return_value = {
        "passages": ["p" * 500], "citations": [], "top_confidence": 0.70,
    }
    with patch("agent_customer_support.agents.knowledge.KnowledgeAgent._grade",
               new=AsyncMock(return_value=True)), \
         patch("agent_customer_support.agents.knowledge.complete_text",
               return_value="Trong phần mềm gọi là Y, làm thế này."):
        res = await KnowledgeAgent().run(ctx)
    assert res.resolved is True
    assert "Y" in res.reply


async def test_no_answer_marker_reformulates_then_logs():
    ctx = _ctx("hỏi linh tinh")
    ctx.rag.search.return_value = {
        "passages": ["p" * 500], "citations": [], "top_confidence": 0.90,
    }
    # compose returns no_answer both times -> reformulate once -> log_request
    with patch("agent_customer_support.agents.knowledge.KnowledgeAgent._reformulate",
               new=AsyncMock(return_value="reworded")), \
         patch("agent_customer_support.agents.knowledge.complete_text",
               return_value="Không có. [[no_answer]]"):
        res = await KnowledgeAgent().run(ctx)
    assert res.resolved is False
    assert ctx.rag.search.await_count == 2          # original + reformulated
    ctx.backlog.add.assert_awaited_once()
    assert ctx.backlog.add.call_args.kwargs["type"] == "how_to_missing"


async def test_suspected_bug_marker_sets_flag():
    ctx = _ctx("tính năng A bị lỗi")
    ctx.rag.search.return_value = {
        "passages": ["p" * 500], "citations": [], "top_confidence": 0.90,
    }
    with patch("agent_customer_support.agents.knowledge.complete_text",
               return_value="Đáng lẽ chạy. [[suspected_bug:xet-nghiem]]"):
        res = await KnowledgeAgent().run(ctx)
    assert res.suspected_bug is True
    assert res.evidence["module"] == "xet-nghiem"
    assert "[[suspected_bug" not in res.reply
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/agents/test_knowledge.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

Create `agent_customer_support/agents/knowledge.py`:

```python
import re

from agent_customer_support.agents.context import TurnContext
from agent_customer_support.agents.prompts import (
    KNOWLEDGE_GRADER_PROMPT, KNOWLEDGE_REFORMULATE_PROMPT, KNOWLEDGE_COMPOSE_PROMPT,
)
from agent_customer_support.config import get_settings
from agent_customer_support.llm import complete_text
from agent_customer_support.models import AgentResult

_NO_ANSWER_RE = re.compile(r"\[\[no_answer\]\]")
_BUG_RE = re.compile(r"\[\[suspected_bug:([a-zA-Z0-9_\-]+)\]\]")

HIGH = 0.80
LOW = 0.50
MIN_SUBSTANTIAL_CHARS = 200   # high score but shorter than this => grade


def needs_grading(top_confidence: float, passages: list[str]) -> bool:
    if not passages:
        return False
    if LOW <= top_confidence < HIGH:
        return True
    if top_confidence >= HIGH:
        total = sum(len(p) for p in passages)
        return total < MIN_SUBSTANTIAL_CHARS or len(passages) < 2
    return False  # low score: skip grader, reformulate instead


def parse_markers(text: str) -> tuple[str, str | None, str | None]:
    """Return (clean_text, kind, module) where kind in {None,'no_answer','suspected_bug'}."""
    bug = _BUG_RE.search(text or "")
    if bug:
        clean = _BUG_RE.sub("", text).strip()
        return clean, "suspected_bug", bug.group(1)
    if _NO_ANSWER_RE.search(text or ""):
        return _NO_ANSWER_RE.sub("", text).strip(), "no_answer", None
    return (text or "").strip(), None, None


def _passages_block(passages: list[str]) -> str:
    return "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages))


class KnowledgeAgent:
    name = "knowledge"

    async def _grade(self, question: str, passages: list[str]) -> bool:
        raw = complete_text(
            messages=[{"role": "user",
                       "content": f"CÂU HỎI: {question}\n\nĐOẠN TRÍCH:\n{_passages_block(passages)}"}],
            system=KNOWLEDGE_GRADER_PROMPT,
        )
        import json
        try:
            return bool(json.loads(raw).get("answer_present"))
        except (json.JSONDecodeError, TypeError):
            return False  # fail-closed: don't answer if grader is unparseable

    async def _reformulate(self, ctx: TurnContext, passages: list[str]) -> str:
        hint = ", ".join(ctx.customer.enabled_modules)
        notes = ctx.customer.config_notes or ""
        raw = complete_text(
            messages=[{"role": "user",
                       "content": f"CÂU HỎI GỐC: {ctx.message}\nMODULE: {hint}\nGHI CHÚ: {notes}"}],
            system=KNOWLEDGE_REFORMULATE_PROMPT,
        )
        return (raw or ctx.message).strip()

    async def _compose(self, question: str, passages: list[str]) -> str:
        return complete_text(
            messages=[{"role": "user",
                       "content": f"CÂU HỎI: {question}\n\nĐOẠN TRÍCH:\n{_passages_block(passages)}"}],
            system=KNOWLEDGE_COMPOSE_PROMPT,
        )

    async def _present(self, ctx: TurnContext, passages: list[str], conf: float) -> bool:
        if not passages:
            return False
        if needs_grading(conf, passages):
            return await self._grade(ctx.message, passages)
        return conf >= HIGH  # trust high, distrust low

    async def run(self, ctx: TurnContext) -> AgentResult:
        # 1. initial search
        res = await ctx.rag.search(ctx.message,
                                   collection=get_settings().product_collection)
        passages = res.get("passages", []) or []
        conf = res.get("top_confidence", 0.0)
        citations = res.get("citations", []) or []

        present = await self._present(ctx, passages, conf)

        # 2. reformulate once on miss
        if not present:
            new_query = await self._reformulate(ctx, passages)
            res = await ctx.rag.search(new_query,
                                       collection=get_settings().product_collection)
            passages = res.get("passages", []) or []
            conf = res.get("top_confidence", 0.0)
            citations = res.get("citations", []) or []
            present = await self._present(ctx, passages, conf)

        # 3. still nothing -> log + unresolved
        if not present:
            await ctx.backlog.add(
                customer_id=ctx.customer.customer_id, type="how_to_missing",
                summary=ctx.message, module=None, transcript=ctx.transcript)
            return AgentResult(
                reply="Mình chưa tìm thấy thông tin cụ thể này trong tài liệu. "
                      "Đã ghi nhận để đội hỗ trợ bổ sung.",
                resolved=False, citations=citations)

        # 4. compose grounded answer (compose may still emit no_answer / suspected_bug)
        composed = await self._compose(ctx.message, passages)
        clean, kind, module = parse_markers(composed)

        if kind == "no_answer":
            await ctx.backlog.add(
                customer_id=ctx.customer.customer_id, type="how_to_missing",
                summary=ctx.message, module=None, transcript=ctx.transcript)
            return AgentResult(
                reply="Mình chưa tìm thấy thông tin cụ thể này trong tài liệu. "
                      "Đã ghi nhận để đội hỗ trợ bổ sung.",
                resolved=False, citations=citations)

        if kind == "suspected_bug":
            return AgentResult(reply=clean, resolved=False, suspected_bug=True,
                               evidence={"module": module, "summary": ctx.message},
                               citations=citations)

        return AgentResult(reply=clean, resolved=True, citations=citations)
```

> **Note for the implementer:** `_reformulate` is patched in `test_no_answer_marker_reformulates_then_logs`; in that test `complete_text` always returns `[[no_answer]]`, so both the first compose (after a trusted high score) and the post-reformulation compose return no_answer → exactly two `rag.search` calls → `log_request`. `test_log_request` flow uses `type="how_to_missing"` which the existing `RequestRecord`/`backlog.add` already accept.

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/agents/test_knowledge.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent_customer_support/agents/knowledge.py tests/agents/test_knowledge.py
git commit -m "feat: KnowledgeAgent orchestrated pipeline with hybrid grader + reformulation"
```

---

### Task 12: FlowAgent (flow tools + goto state machine)

**Files:**
- Create: `agent_customer_support/agents/flow.py`
- Test: `tests/agents/test_flow.py`

Owns the flow tool loop (`list_flows`, `get_flow`) + the `[[goto:...]]` advancement that old `core.py` did. Reuses `FlowEngine` and `build_system_prompt`.

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_flow.py`:

```python
import pytest
from unittest.mock import patch, AsyncMock
from agent_customer_support.agents.flow import FlowAgent, parse_goto
from agent_customer_support.agents.context import TurnContext
from agent_customer_support.models import (
    CustomerProfile, SessionState, Conversation, Flow, FlowStep, FlowTransition,
    FlowOutcome,
)

pytestmark = pytest.mark.asyncio


def _flow():
    return Flow(id="f1", title="T", module="m", steps=[
        FlowStep(id="s1", say="bước 1",
                 next=[FlowTransition(when="xong", goto="s2")]),
        FlowStep(id="s2", say="bước 2",
                 next=[FlowTransition(when="ok", goto="done")]),
    ], outcomes={"done": FlowOutcome(type="success", say="Hoàn tất")})


def _ctx(session, flow_store) -> TurnContext:
    return TurnContext(
        customer=CustomerProfile(customer_id="c1", name="C1", enabled_modules=["m"]),
        session=session,
        conversation=Conversation(conversation_id="cv1", customer_id="c1"),
        message="xong rồi", transcript="user: xong rồi",
        rag=AsyncMock(), flow_store=flow_store, backlog=AsyncMock(),
        escalator=AsyncMock(),
    )


def test_parse_goto():
    assert parse_goto("tiếp tục [[goto:s2]]") == ("tiếp tục", "s2")


async def test_advances_active_flow_step():
    fs = AsyncMock()
    fs.get.return_value = _flow()
    session = SessionState(conversation_id="cv1", active_flow_id="f1",
                           current_step_id="s1")
    seq = [{"stop_reason": "end_turn", "text": "Làm bước 2 nhé [[goto:s2]]",
            "tool_calls": []}]
    with patch("agent_customer_support.agents.flow.complete_with_tools",
               side_effect=seq):
        res = await FlowAgent().run(_ctx(session, fs))
    assert res.new_session.current_step_id == "s2"
    assert "[[goto" not in res.reply


async def test_outcome_clears_session():
    fs = AsyncMock()
    fs.get.return_value = _flow()
    session = SessionState(conversation_id="cv1", active_flow_id="f1",
                           current_step_id="s2")
    seq = [{"stop_reason": "end_turn", "text": "Tốt [[goto:done]]", "tool_calls": []}]
    with patch("agent_customer_support.agents.flow.complete_with_tools",
               side_effect=seq):
        res = await FlowAgent().run(_ctx(session, fs))
    assert res.new_session.active_flow_id is None
    assert res.new_session.current_step_id is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/agents/test_flow.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

Create `agent_customer_support/agents/flow.py`:

```python
import json
import re

from agent_customer_support.agents.context import TurnContext
from agent_customer_support.config import get_settings
from agent_customer_support.flows.engine import FlowEngine
from agent_customer_support.llm import complete_with_tools
from agent_customer_support.models import AgentResult
from agent_customer_support.agent.prompt import build_system_prompt
from agent_customer_support.agent.tools import ToolContext, dispatch

_GOTO_RE = re.compile(r"\[\[goto:([a-zA-Z0-9_\-]+)\]\]")
FLOW_TOOLS = [
    {"name": "list_flows",
     "description": "Liệt kê các quy trình khả dụng cho khách hàng hiện tại.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_flow",
     "description": "Lấy chi tiết một quy trình theo flow_id để dẫn từng bước.",
     "input_schema": {"type": "object",
                      "properties": {"flow_id": {"type": "string"}},
                      "required": ["flow_id"]}},
]
MAX_ROUNDS = 5


def parse_goto(text: str) -> tuple[str, str | None]:
    m = _GOTO_RE.search(text or "")
    if not m:
        return (text or "", None)
    return (_GOTO_RE.sub("", text).strip(), m.group(1))


class FlowAgent:
    name = "flow"

    async def run(self, ctx: TurnContext) -> AgentResult:
        session = ctx.session.model_copy(deep=True)
        active_flow = None
        if session.active_flow_id:
            active_flow = await ctx.flow_store.get(session.active_flow_id)

        system = build_system_prompt(ctx.customer, session, active_flow)
        tool_ctx = ToolContext(
            customer=ctx.customer, rag=ctx.rag, flow_store=ctx.flow_store,
            backlog=ctx.backlog, escalator=ctx.escalator,
            conversation_id=session.conversation_id, transcript=ctx.transcript,
        )
        messages = [{"role": "user", "content": ctx.message}]
        final_text = ""
        is_anthropic = "claude" in get_settings().agent_model

        for _ in range(MAX_ROUNDS):
            out = complete_with_tools(messages=messages, tools=FLOW_TOOLS, system=system)
            if out["stop_reason"] != "tool_use":
                final_text = out.get("text") or ""
                break
            if is_anthropic:
                messages.append({"role": "assistant", "content": out.get("text") or ""})
                results = []
                for call in out["tool_calls"]:
                    r = await dispatch(call["name"], call["input"], tool_ctx)
                    results.append({"type": "tool_result", "tool_use_id": call["id"],
                                    "content": json.dumps(r, ensure_ascii=False)})
                messages.append({"role": "user", "content": results})
            else:
                messages.append({"role": "assistant", "content": out.get("text"),
                                 "tool_calls": [{"id": c["id"], "type": "function",
                                                 "function": {"name": c["name"],
                                                 "arguments": json.dumps(c["input"])}}
                                                for c in out["tool_calls"]]})
                for call in out["tool_calls"]:
                    r = await dispatch(call["name"], call["input"], tool_ctx)
                    messages.append({"role": "tool", "tool_call_id": call["id"],
                                     "content": json.dumps(r, ensure_ascii=False)})

        clean, goto = parse_goto(final_text)
        effective = active_flow or tool_ctx.last_fetched_flow
        escalated = False
        if effective and goto:
            res = FlowEngine.resolve(effective, goto)
            if res.kind == "outcome":
                if res.outcome and res.outcome.type == "escalate":
                    await ctx.escalator.escalate(
                        customer_id=ctx.customer.customer_id,
                        reason=res.outcome.reason or "flow escalate",
                        transcript=ctx.transcript)
                    escalated = True
                session.active_flow_id = None
                session.current_step_id = None
            elif res.step is not None:
                if not session.active_flow_id:
                    session.active_flow_id = effective.id
                session.current_step_id = res.step.id

        return AgentResult(reply=clean, new_session=session, escalated=escalated)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/agents/test_flow.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent_customer_support/agents/flow.py tests/agents/test_flow.py
git commit -m "feat: add FlowAgent with goto state machine"
```

---

### Task 13: IssueVerificationAgent (evidence gathering, multimodal)

**Files:**
- Create: `agent_customer_support/agents/verification.py`
- Test: `tests/agents/test_verification.py`

Builds a multimodal user message (text + image blocks) so screenshots reach the model. Parses `[[evidence_ready]]` → `evidence_complete=True`.

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_verification.py`:

```python
import pytest
from unittest.mock import patch, AsyncMock
from agent_customer_support.agents.verification import IssueVerificationAgent
from agent_customer_support.agents.context import TurnContext
from agent_customer_support.models import (
    CustomerProfile, SessionState, Conversation, Attachment,
)

pytestmark = pytest.mark.asyncio


def _ctx(message, attachments=None) -> TurnContext:
    s = SessionState(conversation_id="cv1", pending="verify_issue",
                     pending_context={"module": "m", "summary": "A bị lỗi"})
    return TurnContext(
        customer=CustomerProfile(customer_id="c1", name="C1"),
        session=s,
        conversation=Conversation(conversation_id="cv1", customer_id="c1"),
        message=message, attachments=attachments or [],
        transcript=f"user: {message}",
        rag=AsyncMock(), backlog=AsyncMock(), escalator=AsyncMock(),
    )


async def test_insufficient_evidence_asks_more():
    with patch("agent_customer_support.agents.verification.complete_with_tools",
               return_value={"stop_reason": "end_turn",
                             "text": "Bạn gửi giúp ảnh lỗi nhé?", "tool_calls": []}):
        res = await IssueVerificationAgent().run(_ctx("nó cứ lỗi thôi"))
    assert res.evidence_complete is False
    assert "ảnh" in res.reply


async def test_evidence_ready_marks_complete():
    att = Attachment(kind="image", media_type="image/png", data="QUJD")
    with patch("agent_customer_support.agents.verification.complete_with_tools",
               return_value={"stop_reason": "end_turn",
                             "text": "Đã nhận đủ thông tin. [[evidence_ready]]",
                             "tool_calls": []}):
        res = await IssueVerificationAgent().run(_ctx("đây là ảnh", [att]))
    assert res.evidence_complete is True
    assert "[[evidence_ready]]" not in res.reply
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/agents/test_verification.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

Create `agent_customer_support/agents/verification.py`:

```python
import re

from agent_customer_support.agents.context import TurnContext
from agent_customer_support.agents.prompts import VERIFICATION_PROMPT
from agent_customer_support.config import get_settings
from agent_customer_support.llm import complete_with_tools
from agent_customer_support.llm.normalize import (
    to_anthropic_content, to_openai_content,
)
from agent_customer_support.models import AgentResult

_READY_RE = re.compile(r"\[\[evidence_ready\]\]")


class IssueVerificationAgent:
    name = "verification"

    async def run(self, ctx: TurnContext) -> AgentResult:
        is_anthropic = "claude" in get_settings().agent_model
        if is_anthropic:
            content = to_anthropic_content(ctx.message, ctx.attachments)
        else:
            content = to_openai_content(ctx.message, ctx.attachments)

        out = complete_with_tools(
            messages=[{"role": "user", "content": content}],
            tools=[], system=VERIFICATION_PROMPT,
        )
        text = out.get("text") or ""
        ready = bool(_READY_RE.search(text))
        clean = _READY_RE.sub("", text).strip()

        if ready:
            evidence = dict(ctx.session.pending_context or {})
            evidence["has_image"] = any(a.kind == "image" for a in ctx.attachments)
            return AgentResult(reply=clean, evidence_complete=True, evidence=evidence)
        return AgentResult(reply=clean, evidence_complete=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/agents/test_verification.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent_customer_support/agents/verification.py tests/agents/test_verification.py
git commit -m "feat: add IssueVerificationAgent with multimodal evidence"
```

---

## Phase 3 — Coordinator

### Task 14: Coordinator orchestration

**Files:**
- Create: `agent_customer_support/agents/coordinator.py`
- Test: `tests/agents/test_coordinator.py`

The coordinator wires stores + agents, loads context, runs the §5 turn flow, persists. Agents are injected so tests stub them.

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_coordinator.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from agent_customer_support.agents.coordinator import Coordinator
from agent_customer_support.models import (
    AgentResult, CustomerProfile, SessionState, Conversation,
)

pytestmark = pytest.mark.asyncio


def _coord():
    c = Coordinator()
    c.customers = AsyncMock()
    c.customers.get.return_value = CustomerProfile(customer_id="c1", name="C1",
                                                   enabled_modules=["m"])
    c.conversations = AsyncMock()
    c.conversations.load.return_value = Conversation(conversation_id="cv1",
                                                     customer_id="c1")
    c.sessions = AsyncMock()
    c.sessions.get.return_value = SessionState(conversation_id="cv1")
    c.rag = AsyncMock(); c.flow_store = AsyncMock()
    c.backlog = AsyncMock(); c.escalator = AsyncMock()
    # agent stubs
    c.guardrail = MagicMock()
    c.guardrail.check_input = AsyncMock(return_value={"pass": True, "reason": ""})
    c.guardrail.check_output = AsyncMock(return_value={"pass": True, "reason": ""})
    c.triage = MagicMock(); c.knowledge = MagicMock()
    c.flow = MagicMock(); c.verification = MagicMock(); c.escalation = MagicMock()
    return c


async def test_input_guardrail_block_short_circuits():
    c = _coord()
    c.guardrail.check_input = AsyncMock(return_value={"pass": False, "reason": "empty"})
    res = await c.handle_turn(customer_id="c1", conversation_id="cv1",
                              message="  ", attachments=[])
    assert res.escalated is False
    c.triage.run = AsyncMock()  # never called
    c.triage.run.assert_not_called()


async def test_triage_clarify_returns_question():
    c = _coord()
    c.triage.run = AsyncMock(return_value=AgentResult(action="reply",
                                                      reply="Bạn cần gì?"))
    res = await c.handle_turn(customer_id="c1", conversation_id="cv1",
                              message="?", attachments=[])
    assert res.reply == "Bạn cần gì?"


async def test_knowledge_resolved_returns_reply():
    c = _coord()
    c.triage.run = AsyncMock(return_value=AgentResult(action="route",
                                                      routed_to="knowledge"))
    c.knowledge.run = AsyncMock(return_value=AgentResult(reply="đáp án",
                                                         resolved=True))
    res = await c.handle_turn(customer_id="c1", conversation_id="cv1",
                              message="cách làm X", attachments=[])
    assert res.reply == "đáp án"


async def test_knowledge_unresolved_escalates():
    c = _coord()
    c.triage.run = AsyncMock(return_value=AgentResult(action="route",
                                                      routed_to="knowledge"))
    c.knowledge.run = AsyncMock(return_value=AgentResult(reply="", resolved=False))
    c.escalation.run = AsyncMock(return_value=AgentResult(reply="chuyển nhân viên",
                                                          escalated=True))
    res = await c.handle_turn(customer_id="c1", conversation_id="cv1",
                              message="lỗi lạ", attachments=[])
    assert res.escalated is True
    c.escalation.run.assert_awaited_once()


async def test_suspected_bug_starts_verification():
    c = _coord()
    c.triage.run = AsyncMock(return_value=AgentResult(action="route",
                                                      routed_to="knowledge"))
    c.knowledge.run = AsyncMock(return_value=AgentResult(
        reply="nghi lỗi", resolved=False, suspected_bug=True,
        evidence={"module": "m", "summary": "A lỗi"}))
    c.verification.run = AsyncMock(return_value=AgentResult(
        reply="gửi ảnh giúp mình", evidence_complete=False))
    res = await c.handle_turn(customer_id="c1", conversation_id="cv1",
                              message="A bị lỗi", attachments=[])
    assert "ảnh" in res.reply
    saved = c.sessions.save.call_args.args[0]
    assert saved.pending == "verify_issue"


async def test_pending_verification_resumes_and_escalates_when_complete():
    c = _coord()
    c.sessions.get.return_value = SessionState(
        conversation_id="cv1", pending="verify_issue",
        pending_context={"module": "m", "summary": "A lỗi"})
    c.verification.run = AsyncMock(return_value=AgentResult(
        reply="đã đủ", evidence_complete=True,
        evidence={"module": "m", "summary": "A lỗi", "has_image": True}))
    c.escalation.run = AsyncMock(return_value=AgentResult(reply="chuyển nhân viên",
                                                          escalated=True))
    res = await c.handle_turn(customer_id="c1", conversation_id="cv1",
                              message="đây là ảnh", attachments=[])
    assert res.escalated is True
    c.backlog.add.assert_awaited_once()
    saved = c.sessions.save.call_args.args[0]
    assert saved.pending is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/agents/test_coordinator.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

Create `agent_customer_support/agents/coordinator.py`:

```python
from agent_customer_support.agents.context import TurnContext
from agent_customer_support.agents.escalation import EscalationAgent
from agent_customer_support.agents.flow import FlowAgent
from agent_customer_support.agents.guardrail import GuardrailAgent
from agent_customer_support.agents.knowledge import KnowledgeAgent
from agent_customer_support.agents.triage import TriageAgent
from agent_customer_support.agents.verification import IssueVerificationAgent
from agent_customer_support.escalation import Escalator
from agent_customer_support.models import (
    AgentResult, ChatResponse, CustomerProfile, Turn,
)
from agent_customer_support.rag_client import RagClient
from agent_customer_support.stores.conversation_store import ConversationStore
from agent_customer_support.stores.customer_registry import CustomerRegistry
from agent_customer_support.stores.flow_store import FlowStore
from agent_customer_support.stores.request_backlog import RequestBacklog
from agent_customer_support.stores.session_store import SessionStore

_BLOCK_REPLY = "Xin lỗi, mình chưa thể xử lý nội dung này. Bạn vui lòng nhập câu hỏi về phần mềm CenLab nhé."
_FALLBACK_REPLY = "Xin lỗi, mình cần kiểm tra lại thông tin này. Bạn vui lòng thử lại hoặc yêu cầu gặp nhân viên hỗ trợ."


class Coordinator:
    def __init__(self) -> None:
        self.customers = CustomerRegistry()
        self.conversations = ConversationStore()
        self.flow_store = FlowStore()
        self.backlog = RequestBacklog()
        self.sessions = SessionStore()
        self.rag = RagClient()
        self.escalator = Escalator()
        self.guardrail = GuardrailAgent()
        self.triage = TriageAgent()
        self.knowledge = KnowledgeAgent()
        self.flow = FlowAgent()
        self.verification = IssueVerificationAgent()
        self.escalation = EscalationAgent()

    async def handle_turn(self, *, customer_id: str, conversation_id: str,
                          message: str, attachments: list) -> ChatResponse:
        # 1. Load context
        customer = await self.customers.get(customer_id) or CustomerProfile(
            customer_id=customer_id, name=customer_id)
        session = await self.sessions.get(conversation_id)
        conv = await self.conversations.load(conversation_id)
        transcript = "\n".join(f"{t.role}: {t.content}" for t in conv.turns)
        ctx = TurnContext(
            customer=customer, session=session, conversation=conv,
            message=message, attachments=attachments,
            transcript=transcript + f"\nuser: {message}",
            rag=self.rag, flow_store=self.flow_store,
            backlog=self.backlog, escalator=self.escalator,
        )

        # 2. Input guardrail
        gin = await self.guardrail.check_input(message)
        if not gin["pass"]:
            return await self._finish(ctx, AgentResult(reply=_BLOCK_REPLY),
                                      session)

        # 3-6. Route
        result = await self._route(ctx, session)

        # 7. Output guardrail
        gout = await self.guardrail.check_output(result.reply)
        if not gout["pass"]:
            result = AgentResult(reply=_FALLBACK_REPLY,
                                 escalated=result.escalated,
                                 new_session=result.new_session)

        return await self._finish(ctx, result, session)

    async def _route(self, ctx: TurnContext, session) -> AgentResult:
        # Resume pending verification
        if session.pending == "verify_issue":
            res = await self.verification.run(ctx)
            return await self._after_verification(ctx, res, session)

        # Active flow
        if session.active_flow_id:
            return await self.flow.run(ctx)

        # Triage
        tri = await self.triage.run(ctx)
        if tri.action == "reply":
            return tri
        if tri.routed_to == "flow":
            return await self.flow.run(ctx)
        if tri.routed_to == "escalate":
            return await self.escalation.run(ctx, reason="user requested human")

        # knowledge
        kn = await self.knowledge.run(ctx)
        if kn.suspected_bug:
            session.pending = "verify_issue"
            session.pending_context = kn.evidence
            ver = await self.verification.run(ctx)
            return await self._after_verification(ctx, ver, session)
        if kn.resolved is False:
            return await self.escalation.run(ctx, reason="knowledge unresolved")
        return kn

    async def _after_verification(self, ctx, res: AgentResult, session) -> AgentResult:
        if not res.evidence_complete:
            return res  # keep pending; coordinator persists session as-is
        # evidence ready → log bug + escalate
        ev = res.evidence or {}
        await self.backlog.add(
            customer_id=ctx.customer.customer_id, type="bug",
            summary=ev.get("summary", "bug"), module=ev.get("module"),
            transcript=ctx.transcript)
        session.pending = None
        session.pending_context = None
        esc = await self.escalation.run(ctx, reason="verified bug")
        esc.new_session = session
        return esc

    async def _finish(self, ctx: TurnContext, result: AgentResult,
                      session) -> ChatResponse:
        # Apply session changes
        new_session = result.new_session or session
        # set pending if route started verification
        new_session.conversation_id = ctx.session.conversation_id
        if session.pending == "verify_issue" and result.new_session is None:
            new_session.pending = session.pending
            new_session.pending_context = session.pending_context
        await self.sessions.save(new_session)

        # Persist turns
        await self.conversations.append(
            ctx.session.conversation_id, ctx.customer.customer_id,
            Turn(role="user", content=ctx.message, attachments=ctx.attachments))
        await self.conversations.append(
            ctx.session.conversation_id, ctx.customer.customer_id,
            Turn(role="assistant", content=result.reply))
        return ChatResponse(conversation_id=ctx.session.conversation_id,
                            reply=result.reply, escalated=result.escalated,
                            citations=result.citations)
```

> **Note for the implementer:** the `session.pending` propagation in `_finish` is subtle — the route that *starts* verification mutates `session` directly (sets `pending`), and `_route` returns the verification's `AgentResult` (whose `new_session` is None when still gathering evidence). The two coordinator tests `test_suspected_bug_starts_verification` and `test_pending_verification_resumes...` pin this behavior. If they pass, the propagation is correct.

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/agents/test_coordinator.py -v`
Expected: PASS. If the pending-propagation tests fail, adjust `_finish`/`_route` so that (a) starting verification persists `pending="verify_issue"`, and (b) completing it persists `pending=None`.

- [ ] **Step 5: Commit**

```bash
git add agent_customer_support/agents/coordinator.py tests/agents/test_coordinator.py
git commit -m "feat: add Coordinator orchestration"
```

---

## Phase 4 — Wiring & Cleanup

### Task 15: Widget endpoint passes attachments to Coordinator

**Files:**
- Modify: `agent_customer_support/channels/widget.py`
- Modify: `agent_customer_support/server.py` (import path for `get_agent`)
- Test: `tests/channels/test_widget.py`

- [ ] **Step 1: Update the failing test**

Replace the body of `tests/channels/test_widget.py` with (adjust to existing fixtures as needed):

```python
import pytest
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient
from agent_customer_support.server import app
from agent_customer_support.channels.widget import get_agent
from agent_customer_support.models import ChatResponse

pytestmark = pytest.mark.asyncio


def test_chat_passes_attachments():
    fake = AsyncMock()
    fake.handle_turn.return_value = ChatResponse(conversation_id="cv1", reply="hi")
    app.dependency_overrides[get_agent] = lambda: fake
    client = TestClient(app)
    resp = client.post("/widget/chat", json={
        "customer_id": "c1", "conversation_id": "cv1", "message": "hello",
        "attachments": [{"kind": "image", "media_type": "image/png", "data": "QUJD"}],
    })
    assert resp.status_code == 200
    assert resp.json()["reply"] == "hi"
    kwargs = fake.handle_turn.call_args.kwargs
    assert kwargs["message"] == "hello"
    assert len(kwargs["attachments"]) == 1
    app.dependency_overrides.clear()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/channels/test_widget.py -v`
Expected: FAIL — `get_agent` still returns `AgentCore`; `handle_turn` lacks `attachments`.

- [ ] **Step 3: Write minimal implementation**

Replace `agent_customer_support/channels/widget.py`:

```python
from fastapi import APIRouter, Depends
from agent_customer_support.models import ChatRequest, ChatResponse
from agent_customer_support.agents.coordinator import Coordinator

router = APIRouter(prefix="/widget", tags=["widget"])


def get_agent() -> Coordinator:
    return Coordinator()


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, agent: Coordinator = Depends(get_agent)) -> ChatResponse:
    return await agent.handle_turn(
        customer_id=req.customer_id,
        conversation_id=req.conversation_id,
        message=req.message,
        attachments=req.attachments,
    )
```

`server.py` re-exports `get_agent` from widget — no change needed since it already imports from `channels.widget`.

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/channels/test_widget.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent_customer_support/channels/widget.py tests/channels/test_widget.py
git commit -m "feat: wire widget endpoint to Coordinator with attachments"
```

---

### Task 16: Delete old `agent/core.py` and its tests

**Files:**
- Delete: `agent_customer_support/agent/core.py`
- Delete/Move: `tests/agent/test_core.py`

The `parse_goto` tests in `test_core.py` are now covered by `tests/agents/test_flow.py`. Behavior is preserved by the new agents.

- [ ] **Step 1: Remove the files**

```bash
git rm agent_customer_support/agent/core.py tests/agent/test_core.py
```

- [ ] **Step 2: Find dangling references**

Run: `grep -rn "agent.core\|AgentCore" agent_customer_support tests scripts`
Expected: no results. If any remain (e.g., `scripts/smoke_chat.py`), update them to import `Coordinator` and call `handle_turn(..., message=..., attachments=[])`.

- [ ] **Step 3: Run the full suite**

Run: `poetry run pytest -q`
Expected: PASS (all tests green).

- [ ] **Step 4: Lint/type check**

Run: `poetry run ruff check . && poetry run mypy agent_customer_support`
Expected: clean (fix any import/type issues surfaced).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: remove monolithic AgentCore (replaced by Coordinator)"
```

---

### Task 17: Frontend — image upload in InputBar

**Files:**
- Modify: `ui/components/InputBar.tsx`
- Modify: `ui/lib/api.ts`

Adds a file-picker that reads an image to base64 and includes it in the `attachments` array of the chat POST.

- [ ] **Step 1: Update the API client**

In `ui/lib/api.ts`, extend the request type and send `attachments`. Add:

```ts
export type Attachment = {
  kind: "image";
  media_type: string;
  data: string; // base64 (no data: prefix)
};

export async function sendMessage(
  customerId: string,
  conversationId: string,
  message: string,
  attachments: Attachment[] = []
) {
  const res = await fetch(`${API_BASE}/widget/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      customer_id: customerId,
      conversation_id: conversationId,
      message,
      attachments,
    }),
  });
  if (!res.ok) throw new Error(`chat failed: ${res.status}`);
  return res.json();
}
```

(Adapt `API_BASE`/signature names to the existing file — keep the existing exports working.)

- [ ] **Step 2: Add the upload control in InputBar**

In `ui/components/InputBar.tsx`, add a hidden file input + button that converts the selected image to base64 and stages it as an `Attachment` passed to `sendMessage`:

```tsx
async function fileToBase64(file: File): Promise<string> {
  const buf = await file.arrayBuffer();
  let binary = "";
  const bytes = new Uint8Array(buf);
  for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

// in component state:
const [pending, setPending] = useState<Attachment[]>([]);

async function onPickImage(e: React.ChangeEvent<HTMLInputElement>) {
  const file = e.target.files?.[0];
  if (!file) return;
  const data = await fileToBase64(file);
  setPending((p) => [...p, { kind: "image", media_type: file.type, data }]);
}

// on send: pass `pending` to sendMessage(...), then setPending([])
```

Wire the existing send handler to pass `pending` and clear it after send. Render a small thumbnail/filename chip for staged attachments.

- [ ] **Step 3: Manual verification**

Run the UI dev server and backend, attach a PNG, send a "feature broken" message, and confirm the POST body includes `attachments[0].data`.

Run: `cd ui && npm run dev` (and the backend per `docs/DEV.md`)
Expected: network tab shows attachments in the request; verification flow asks for/accepts the image.

- [ ] **Step 4: Build check**

Run: `cd ui && npm run build`
Expected: type-checks and builds clean.

- [ ] **Step 5: Commit**

```bash
git add ui/components/InputBar.tsx ui/lib/api.ts
git commit -m "feat(ui): add image upload for bug evidence"
```

---

### Task 18: Update remaining prompt module + final regression

**Files:**
- Modify: `agent_customer_support/agent/prompt.py` (keep `build_system_prompt`; the old `_BASE` is now superseded by per-agent prompts but `build_system_prompt` is still used by `FlowAgent` for flow context — leave it intact)
- Test: full suite + eval

- [ ] **Step 1: Confirm `build_system_prompt` still used only by FlowAgent**

Run: `grep -rn "build_system_prompt" agent_customer_support`
Expected: referenced in `agents/flow.py`. No change required.

- [ ] **Step 2: Run full test suite**

Run: `poetry run pytest -q`
Expected: all green.

- [ ] **Step 3: Run the eval set (no regression on deflection)**

Run: `poetry run python eval/run_eval.py` (or the documented eval command in `docs/DEV.md`)
Expected: deflection on the "Hướng dẫn sử dụng" set holds vs. the pre-refactor baseline.

- [ ] **Step 4: Lint + type check**

Run: `poetry run ruff check . && poetry run mypy agent_customer_support`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "test: full regression after multi-agent re-architecture"
```

---

## Self-Review Notes (for the implementer)

- **Behavior parity:** Task 11b (KnowledgeAgent, now an orchestrated grader+reformulation pipeline rather than the old model-driven tool loop) and Task 12 (FlowAgent) reproduce the old `core.py` answer/flow paths; Task 11a reframes `grounding_note`. Tasks 9, 10, 13 add new behavior (guardrails, clarify, verification). If the eval set regresses on deflection, check the grader threshold (`needs_grading`) and the compose `[[no_answer]]` guard — too aggressive grading lowers deflection, too lax raises hallucination.
- **Provider branching** lives in one place now (`llm/__init__.py`); agents never branch on provider except when building multimodal content (Task 13).
- **Spec B seam:** nothing in these tasks touches `RagClient`'s interface — the in-repo RAG migration can proceed independently afterward.
```

# Design: Multi-Agent Re-Architecture (Spec A)

- **Date:** 2026-06-12
- **Status:** Design approved (pending implementation plan)
- **Repo:** `agent-customer-support`
- **Supersedes runtime structure of:** `agent/core.py` (the monolithic `AgentCore`)
- **Related:** Spec B — RAG in-repo migration (to be written separately)

---

## 1. Motivation

The current agent runtime is a single `AgentCore` class (`agent/core.py`, ~230 lines) that owns
everything in one turn: loading state, building the system prompt, running the multi-round
LLM tool-use loop, branching Anthropic vs OpenAI message formats, dispatching tools, parsing
`[[goto:...]]` flow markers, advancing the flow state machine, and persisting turns.

The responsibilities are tangled, which makes each one hard to test or change in isolation.

**Goal:** decompose into a **coordinator + small, single-responsibility agents**, each understandable
and testable on its own, communicating through well-defined typed interfaces. The coordinator does
pure orchestration (no LLM). Each agent answers three questions clearly: *what it does, how you call
it, what it depends on.*

This spec also **removes the runtime code dependency on `enterprise_llm_service`** for the agent loop
by vendoring a self-contained LLM client. RAG stays behind the existing `RagClient` HTTP interface
(in-repo RAG migration is Spec B).

---

## 2. Scope

**In Spec A:**
- Coordinator (no LLM) implementing the turn flow in §5.
- Six agents: `GuardrailAgent`, `TriageAgent`, `KnowledgeAgent`, `FlowAgent`,
  `IssueVerificationAgent`, `EscalationAgent`.
- Vendored LLM client (`agent_customer_support/llm/`) supporting **both Anthropic and OpenAI**,
  with tool-use normalization and **multimodal (image) input** — replaces the
  `enterprise_llm_service` import in `llm.py`.
- Image/screenshot evidence support end-to-end: `ChatRequest.attachments`, widget endpoint,
  verification agent multimodal messages, conversation persistence, frontend upload control.
- Per-agent system prompts (split from the single `_BASE` prompt).
- New session state for multi-turn sub-flows (`pending`, `pending_context`).

**Out of scope (Spec B or later):**
- RAG/indexing in-repo migration (embedding + Qdrant clients, in-process `RagService`).
  Agents keep calling RAG through `RagClient` over HTTP. The `RagClient` interface is the seam.
- Offline `scripts/index_kb.py` — keeps importing `enterprise_llm_service` for now.
- Zalo channel, live takeover, automated feedback loop (already out of scope in prior spec).

---

## 3. Design Principles

- **One responsibility per agent.** Each agent does exactly one job and nothing else.
- **Coordinator is dumb glue.** Orchestration is deterministic branching over typed agent results —
  no LLM, no business logic hidden inside it. Every branch is a unit-testable state transition.
- **Sequential hand-off, no agent-to-agent calls.** Agents never call each other. Results flow back
  up to the coordinator, which decides the next step.
- **Surface wording is not a routing signal.** "Bug", "broken", "add feature" never route straight
  to escalation. The user's underlying *goal* is resolved by `KnowledgeAgent` first.
- **Typed results, not bare strings.** Each agent returns an `AgentResult` so the coordinator can
  branch deterministically.

---

## 4. Agent Map

```
Coordinator (no LLM)
  ├── GuardrailAgent          input safety + output quality gate
  ├── TriageAgent             clarify ambiguous intent | route when clear
  ├── KnowledgeAgent          try to solve via RAG; log_request if cannot
  ├── FlowAgent               drive step-by-step guided flows
  ├── IssueVerificationAgent  gather evidence for a suspected bug before escalation
  └── EscalationAgent         human handoff (no LLM)
```

| Agent | Input | Output | LLM? | Depends on |
|---|---|---|---|---|
| **GuardrailAgent** | user message OR draft reply | `{pass, block, reason}` | rules first, LLM fallback | LLM client |
| **TriageAgent** | message + history + session | `clarify(question)` or `route(knowledge\|flow\|escalate)` | LLM | LLM client |
| **KnowledgeAgent** | user goal + customer + history | `AgentResult(reply, resolved, suspected_bug)` | LLM + tools | `RagClient`, `RequestBacklog` |
| **FlowAgent** | message + session + active flow | `AgentResult(reply, new_session)` | LLM + tools | `FlowStore`, `FlowEngine` |
| **IssueVerificationAgent** | message + attachments + pending_context | `AgentResult(reply, evidence_complete, evidence)` | LLM (multimodal) | LLM client |
| **EscalationAgent** | reason + transcript + customer | `AgentResult(reply, escalated=true)` | no LLM | `Escalator`, `RequestBacklog` |

### 4.1 TriageAgent routing logic (hybrid: rules + LLM)

```
if session.active_flow_id        -> route: "flow"        # deterministic, no LLM
elif explicit "talk to a person" -> route: "escalate"    # keyword fast-path
else -> LLM call:
    intent clear     -> route: "knowledge" (default) or "flow"
    intent ambiguous -> clarify: ask ONE focused question
```

TriageAgent **never** routes to escalation on complaint language ("bug", "broken", "add feature").
Those route to `KnowledgeAgent`. Because most first messages are ambiguous, `clarify` is a
first-class outcome: triage re-runs each turn with the added context until intent is clear.

### 4.2 KnowledgeAgent resolution flow

```
1. search_knowledge(user_goal)        # via RagClient
2a. grounding_note == "clarification" -> ask ONE clarifying question (stays in KnowledgeAgent)
2b. confident, docs answer the goal   -> answer -> resolved=true
2c. docs confirm feature SHOULD work but user reports it doesn't
                                      -> suspected_bug=true  (coordinator -> IssueVerificationAgent)
2d. no docs / low confidence          -> log_request(how_to_missing); resolved=false
                                         (coordinator -> EscalationAgent)
```

### 4.3 IssueVerificationAgent

Reached only when `KnowledgeAgent` sets `suspected_bug=true`. Single job: collect evidence
(error message, screenshot/image, repro steps) before a bug is escalated/logged.

`GuardrailAgent` checks are deterministic rules first (off-topic keywords, prompt-leak markers,
empty/oversized input), with an LLM fallback only for nuanced cases (output hallucination/tone).
This keeps the common path fast and unit-testable without an LLM mock.

```
- insufficient evidence -> ask for proof; keep session.pending="verify_issue"
- enough evidence        -> evidence_complete=true + structured evidence
                            (coordinator -> log_request(bug, evidence) -> EscalationAgent)
```

It never decides routing and never does the handoff itself.

### 4.4 EscalationAgent

Reached via exactly two paths: (1) user explicitly asks for a human (TriageAgent fast-path),
or (2) `KnowledgeAgent`/`IssueVerificationAgent` exhausted resolution. No LLM — formats the
handoff, calls `Escalator.escalate()`, returns a user-facing "transferring you to staff" reply.

---

## 5. Turn Flow (Coordinator)

```
Coordinator.handle_turn(customer_id, conversation_id, message, attachments):
  1. Load context: customer, session, conversation history -> TurnContext
  2. GuardrailAgent.check_input(message)
        blocked -> return safe refusal (skip all sub-agents)
  3. Resolve entry agent:
        if session.pending == "verify_issue" -> IssueVerificationAgent
        elif session.active_flow_id          -> FlowAgent
        else                                 -> TriageAgent
  4. TriageAgent result:
        clarify -> reply = question (no sub-agent)
        route   -> dispatch to KnowledgeAgent | FlowAgent | EscalationAgent
  5. KnowledgeAgent result:
        resolved=true   -> use reply
        suspected_bug   -> set session.pending="verify_issue",
                           pending_context={summary, module} -> IssueVerificationAgent
        resolved=false  -> EscalationAgent
  6. IssueVerificationAgent result:
        evidence_complete=false -> reply = ask for more (keep pending)
        evidence_complete=true  -> log_request(bug, evidence); clear pending -> EscalationAgent
  7. GuardrailAgent.check_output(reply)
        flagged -> safe fallback / regenerate
  8. Persist turn(s) + session state
  9. Return ChatResponse
```

The coordinator holds no business logic beyond this branching. `session.pending` is what lets
multi-turn sub-flows (verification) resume on the next turn without the coordinator remembering
anything between requests.

---

## 6. State & Model Changes

```python
# SessionState — add multi-turn sub-flow state
class SessionState(BaseModel):
    conversation_id: str
    active_flow_id: str | None = None
    current_step_id: str | None = None
    pending: Literal["verify_issue"] | None = None   # NEW
    pending_context: dict | None = None              # NEW (suspected-bug summary/module)
    updated_at: datetime = Field(default_factory=_now)

# Attachment — image evidence
class Attachment(BaseModel):
    kind: Literal["image"]
    media_type: str    # image/png | image/jpeg
    data: str          # base64 (URL variant decided in plan)

# ChatRequest — accept attachments
class ChatRequest(BaseModel):
    customer_id: str
    conversation_id: str
    message: str
    attachments: list[Attachment] = Field(default_factory=list)   # NEW

# AgentResult — typed sub-agent output the coordinator branches on
class AgentResult(BaseModel):
    reply: str
    routed_to: str | None = None       # tracing only
    resolved: bool | None = None       # KnowledgeAgent
    suspected_bug: bool = False        # KnowledgeAgent -> verification
    evidence_complete: bool = False    # IssueVerificationAgent -> escalate
    evidence: dict | None = None
    escalated: bool = False
    new_session: "SessionState | None" = None
    citations: list[str] = Field(default_factory=list)
```

`Turn` gains optional attachment references so image evidence persists in the conversation history
(exact shape decided in the plan).

---

## 7. Vendored LLM Client

Replaces `from enterprise_llm_service.llm_inference import ...` in `llm.py`. Same return contract
the agents already expect: `{stop_reason, text, tool_calls}`.

```
agent_customer_support/llm/
  __init__.py            # public: complete_with_tools(), complete_text()
  providers/
    base.py              # LLMProvider protocol -> {stop_reason, text, tool_calls}
    anthropic.py         # Anthropic SDK: tool-use + image content blocks
    openai.py            # OpenAI SDK: tool_calls + image content blocks
  normalize.py           # unify Anthropic <-> OpenAI request/response shapes
```

- Provider selected from `settings.agent_model` (`claude*` -> Anthropic, else OpenAI), preserving
  the existing branching that `core.py` did inline.
- **Multimodal:** the message builder constructs provider-correct image blocks, so
  `IssueVerificationAgent` can pass screenshots to the model. This removes the earlier upstream
  risk ("does enterprise_llm_service support image blocks") — we own it.
- API keys/config read from settings/env (carried over from current setup).

---

## 8. Target File Structure

```
agent_customer_support/
  llm/                          # NEW — vendored, replaces enterprise_llm_service import
    __init__.py
    providers/{base,anthropic,openai}.py
    normalize.py
  agents/                       # NEW — replaces monolithic core.py
    base.py                     # Agent protocol: run(ctx) -> AgentResult
    context.py                  # TurnContext (shared input)
    coordinator.py              # orchestration (no LLM) — §5
    guardrail.py
    triage.py
    knowledge.py
    flow.py
    verification.py
    escalation.py
  agent/
    tools.py                    # KEEP — tool dispatch (grouped per agent as needed)
    prompt.py                   # per-agent prompts (split from one big _BASE)
  flows/engine.py               # KEEP unchanged
  stores/*                      # KEEP unchanged
  rag_client.py                 # KEEP (Spec B swaps internals behind same interface)
  channels/widget.py            # accept attachments, pass to coordinator
  server.py                     # KEEP
  models.py                     # add Attachment, AgentResult; extend SessionState/ChatRequest
```

`agent/core.py` is deleted; `coordinator.py` + the `agents/` package replace it.

---

## 9. Testing Strategy

The whole point of this re-architecture is testability. Each unit is tested in isolation:

- **Coordinator (no LLM):** feed stub agent results, assert the branch taken and session
  transitions (clarify, route, suspected_bug -> verify, evidence_complete -> escalate,
  guardrail block). No LLM mock needed — pure state-machine tests.
- **TriageAgent:** deterministic fast-paths (active flow, "talk to a person") tested without LLM;
  ambiguous-intent clarify path tested with a mocked LLM returning a fixed decision.
- **KnowledgeAgent:** mock `RagClient` to return high/medium/low confidence -> assert
  answer / clarify / suspected_bug / log_request branches.
- **FlowAgent:** reuse existing `FlowEngine` tests; assert `[[goto:...]]` parsing and session
  advancement.
- **IssueVerificationAgent:** assert "ask for more evidence" vs "evidence_complete" with/without
  an image attachment.
- **EscalationAgent:** no LLM — assert `Escalator.escalate()` called with correct reason/transcript.
- **LLM client:** unit-test request/response normalization for Anthropic and OpenAI, including
  image blocks, against recorded fixtures.
- **Integration:** full turn with mocked LLM + mocked `RagClient` for the key scenarios
  (ambiguous -> clarify -> answer; suspected bug -> evidence -> escalate; explicit human request).
- **Eval set:** keep the existing Excel golden-eval harness; no behavioral regression on the
  "Hướng dẫn sử dụng" deflection set.

---

## 10. Migration Notes

- Behavior parity first: the new pipeline must reproduce current happy-path behavior
  (RAG answer, flow guidance, escalation) before new behavior (verification, guardrails) is enabled.
- The vendored LLM client is a drop-in for `llm.py`'s two functions — swap behind the same
  `complete_with_tools` / `complete_text` signatures so agents are provider-agnostic.
- `RagClient` interface is frozen so Spec B can swap HTTP -> in-process with zero agent changes.
- Old debug `_dbg` tracing in `core.py` is replaced by structured per-agent logging.
```

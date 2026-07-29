# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
poetry install

# Start dev infra (DynamoDB Local port 8000 + Redis port 6379)
make infra-up

# Run the API server (port 8800, hot-reload)
make run

# Run all tests
make test

# Run a single test file or test
poetry run pytest tests/agents/test_triage.py -v
poetry run pytest tests/agents/test_triage.py::test_some_case -v

# Lint (ruff format + check + mypy)
make lint
```

Tests use DynamoDB Local + `fakeredis` — no real AWS credentials needed. `conftest.py` stubs all required env vars.

## Architecture

This is a **Vietnamese-language customer support agent** for CenLab cloud software, structured as a multi-agent pipeline behind a FastAPI server.

### Request flow

`POST /widget/chat` → `Coordinator.handle_turn()` → agents in sequence → `ChatResponse`

The `Coordinator` (`agents/coordinator.py`) orchestrates:
1. **Input guardrail** — blocks off-topic/harmful input
2. **Triage** — routes to `flow`, `escalate`, or `knowledge`
3. **Knowledge** — RAG search + LLM answer; may detect a suspected bug
4. **Verification** — multi-turn evidence collection when a bug is suspected (state preserved in `session.pending = "verify_issue"`)
5. **Flow** — walks the user through a step/transition/outcome tree (e.g. account recovery)
6. **Escalation** — posts to Zalo webhook and returns a handoff reply
7. **Output guardrail** — replaces hallucinated/out-of-scope replies

Every agent receives a `TurnContext` (`agents/context.py`) and returns `AgentResult` (`models.py`). The `Agent` protocol (`agents/base.py`) is a structural interface — just `name: str` and `async def run(ctx) -> AgentResult`.

### LLM layer

`llm/__init__.py` exports `complete_with_tools` and `complete_text` — these are the only LLM call sites. Model routing is automatic: model names containing `"claude"` use the Anthropic provider; everything else uses OpenAI. Per-agent model overrides are configured in `Settings` (`config.py`) and accessed via `settings.model_for("triage")` etc.

The OpenAI provider builds request params per model family (`llm/providers/openai_provider.py`): reasoning models (`gpt-5*`, `o1/o3/o4*`) get `max_completion_tokens` + `reasoning_effort` and **no** `temperature`; older models keep `max_tokens` + `temperature=0.5`. The facade resolves the reasoning profile once from `Settings` and passes it down, so providers stay pure functions of their arguments. Effort and the token ceiling are enforced by `ENVIRONMENT` alone — `dev` → `low`/4000, `prod` → `high`/8000 (`_REASONING_EFFORT_BY_ENV` in `config.py`); there is no per-key override by design.

### Storage

| Store | Backend | Purpose |
|---|---|---|
| `SessionStore` | Redis | Turn-to-turn state (`active_flow_id`, `pending`, TTL-based) |
| `ConversationStore` | DynamoDB | Full message history |
| `CustomerRegistry` | DynamoDB | Customer profiles & enabled modules |
| `FlowStore` | DynamoDB | Flow definitions (seeded via `scripts/import_flows.py`) |
| `RequestBacklog` | DynamoDB | Bug/feature/how-to records logged on escalation |

### Flows

`models.py` defines `Flow → FlowStep → FlowTransition → FlowOutcome`. `FlowEngine` (`flows/engine.py`) is a pure stateless resolver — `FlowAgent` uses it to advance `session.current_step_id`. Flow JSON files are seeded from `seeds/flows/` via `scripts/import_flows.py`.

### Observability

All tracing goes through `observability/tracing.py` (the only file that imports `langfuse`). It is a no-op when `LANGFUSE_PUBLIC_KEY` is unset. Spans follow the hierarchy: `turn` → `agent.<name>` → `llm` / `tool.<name>` / `rag.search`.

### Key env vars

See `.env-example`. The important runtime ones:
- `ENVIRONMENT` — `dev` (default) or `prod`; sets the enforced reasoning effort and output token ceiling. Prod deployments must inject it explicitly — the default is `dev`, i.e. `low` effort.
- `AGENT_MODEL` — default model (e.g. `gpt-5.4-mini`, `claude-sonnet-4-6`); per-agent overrides via `TRIAGE_MODEL`, `KNOWLEDGE_MODEL`, `KNOWLEDGE_CONTEXTUALIZE_MODEL`, `VERIFICATION_MODEL`, `FLOW_MODEL`, `GUARDRAIL_MODEL`
- `RAG_BASE_URL` — enterprise-llm-service endpoint for vector search (`/rag/query`)
- `LANGFUSE_*` — optional tracing; leave blank to disable
- `DYNAMODB_ENDPOINT_URL` — set to `http://localhost:8000` for local dev

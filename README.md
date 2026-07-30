# Agent Customer Support

A Vietnamese-language customer support agent for CenLab cloud software, built as a multi-agent pipeline behind a FastAPI server, with a Next.js chat widget/admin UI.

## Architecture

`POST /widget/chat` → `Coordinator.handle_turn()` → agents in sequence → `ChatResponse`

The `Coordinator` (`agent_customer_support/agents/coordinator.py`) orchestrates:

1. **Input guardrail** — blocks off-topic/harmful input
2. **Triage** — routes to `flow`, `escalate`, or `knowledge`
3. **Knowledge** — RAG search + LLM answer; may detect a suspected bug
4. **Verification** — multi-turn evidence collection when a bug is suspected
5. **Flow** — walks the user through a step/transition/outcome tree (e.g. account recovery)
6. **Escalation** — posts to a Zalo webhook and returns a handoff reply
7. **Output guardrail** — replaces hallucinated/out-of-scope replies

### LLM layer

`llm/__init__.py` exports `complete_with_tools` and `complete_text` — the only LLM call sites. Model routing is automatic: model names containing `"claude"` use the Anthropic provider; everything else uses OpenAI. Per-agent model overrides live in `Settings` (`config.py`).

### Storage

| Store | Backend | Purpose |
|---|---|---|
| `SessionStore` | Redis | Turn-to-turn state (`active_flow_id`, `pending`, TTL-based) |
| `ConversationStore` | DynamoDB | Full message history |
| `CustomerRegistry` | DynamoDB | Customer profiles & enabled modules |
| `FlowStore` | DynamoDB | Flow definitions (seeded via `scripts/import_flows.py`) |
| `RequestBacklog` | DynamoDB | Bug/feature/how-to records logged on escalation |

### Flows

`models.py` defines `Flow → FlowStep → FlowTransition → FlowOutcome`. `FlowEngine` (`flows/engine.py`) is a pure stateless resolver; `FlowAgent` uses it to advance `session.current_step_id`. Flow JSON files are seeded from `seeds/flows/` via `scripts/import_flows.py`.

### Observability

All tracing goes through `observability/tracing.py` (the only file that imports `langfuse`). It's a no-op when `LANGFUSE_PUBLIC_KEY` is unset. Spans follow the hierarchy: `turn` → `agent.<name>` → `llm` / `tool.<name>` / `rag.search`.

## Getting started

Requires Python 3.13, [Poetry](https://python-poetry.org/), Docker (for local infra), and Node.js (for the UI).

```bash
# Install backend dependencies
poetry install

# Copy env vars and fill in API keys
cp .env-example .env

# Start dev infra: DynamoDB Local (port 8000) + Redis (port 6379)
make infra-up

# Run the API server (port 8800, hot-reload)
make run
```

In a separate terminal, run the UI:

```bash
cd ui
npm install
npm run dev
```

## Testing & linting

```bash
# Run all tests
make test

# Run a single test file or test
poetry run pytest tests/agents/test_triage.py -v
poetry run pytest tests/agents/test_triage.py::test_some_case -v

# Lint (ruff format + check + mypy)
make lint
```

Tests use DynamoDB Local + `fakeredis` — no real AWS credentials needed. `conftest.py` stubs all required env vars.

## Key environment variables

See `.env-example` for the full list. The important runtime ones:

- `ENVIRONMENT` — `dev` (default) or `prod`. Sets the enforced OpenAI reasoning profile: `dev` → `reasoning_effort=low`, 4000 max output tokens; `prod` → `high`, 8000. There is no per-key override — the Dockerfile ships no `ENV` defaults, so a prod deployment must inject `ENVIRONMENT=prod` in its task definition / compose override or it will silently run at `low` effort.
- `AGENT_MODEL` — default model (e.g. `gpt-5.4-mini`, `claude-sonnet-4-6`); override per agent with `TRIAGE_MODEL`, `KNOWLEDGE_MODEL`, `KNOWLEDGE_CONTEXTUALIZE_MODEL`, `VERIFICATION_MODEL`, `FLOW_MODEL`, `GUARDRAIL_MODEL`
- `QDRANT_ENDPOINT` / `QDRANT_API_KEY` — Qdrant instance backing RAG; `GOOGLE_API_KEY` + `EMBEDDING_MODEL` for query/document embeddings
- `LANGFUSE_*` — optional tracing; leave blank to disable
- `DYNAMODB_ENDPOINT_URL` — set to `http://localhost:8000` for local dev

## Project layout

```
agent_customer_support/
  agent/          # prompt assembly, tool definitions
  agents/         # coordinator + pipeline agents (triage, knowledge, flow, verification, ...)
  channels/       # FastAPI routers (widget, admin)
  flows/          # stateless flow-resolution engine
  llm/            # provider-agnostic LLM facade (Anthropic/OpenAI)
  observability/  # Langfuse tracing
  rag/            # embeddings, QA indexing
  stores/         # Redis/DynamoDB-backed stores
  server.py       # FastAPI app entrypoint
scripts/          # eval, import, and smoke-test scripts
seeds/            # flow definitions seeded into FlowStore
ui/               # Next.js chat widget + admin UI
tests/            # pytest suite mirroring the package layout
```

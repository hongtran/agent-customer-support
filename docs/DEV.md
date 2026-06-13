# Development Guide

## Prerequisites
- Python 3.13 (via pyenv: `pyenv install 3.13.13`)
- Poetry 2.x (`pip install poetry`)
- Docker (for dev infra)

## First-time setup

```bash
# 1. Use Python 3.13
poetry env use "$(pyenv which python3.13)"

# 2. Install dependencies
poetry install

# 3. (Optional) Install enterprise_llm_service — ONLY needed for offline KB indexing
#    (scripts/index_kb.py). The runtime LLM client is vendored; the agent no longer
#    imports enterprise_llm_service.
poetry run pip install --no-deps /path/to/enterprise-llm-service/dist/enterprise_llm_service-1.0.3-py3-none-any.whl

# 4. Start dev infra (DynamoDB Local + Redis)
docker compose up -d

# 5. Copy env file and fill in API keys
cp .env-example .env
# Edit .env: set OPENAI_API_KEY or ANTHROPIC_API_KEY, GOOGLE_API_KEY
```

## Running the agent

```bash
# Seed flows into DynamoDB
poetry run python scripts/import_flows.py seeds/flows

# Start the API server (port 8800)
make run

# Test the chat endpoint
curl -X POST http://localhost:8800/widget/chat \
  -H "Content-Type: application/json" \
  -d '{"customer_id":"ttp","conversation_id":"cv1","message":"Làm sao xử lý PYC sự cố?"}'
```

## Tests

```bash
make test           # run all tests
make lint           # ruff + mypy
```

Tests use DynamoDB Local (port 8000) and fakeredis for session store — no real AWS credentials needed.

## Smoke test (requires real LLM + RAG)

```bash
# enterprise-llm-service must be running at :7799 with cenlab collection indexed
poetry run python scripts/smoke_chat.py
```

## Eval (requires real LLM + indexed KB)

```bash
# Generate golden set from the support Excel file
poetry run python eval/golden_from_excel.py "/path/to/1. Cac yeu cau TTP-Cenlab 2026.xlsx"

# Run triage + deflection eval
poetry run python eval/run_eval.py eval/golden.json
```

## Observability (Langfuse)

The agent emits a hierarchical trace per turn — root span (one conversation turn)
→ per-agent spans (`agent.triage`, `agent.knowledge`, `agent.flow`,
`agent.verification`, `agent.escalation`) → LLM generations (`llm`, with model +
token usage), tool spans (`tool.<name>`), and `rag.search`. Traces are grouped by
`session_id = conversation_id`, so a whole multi-turn conversation (including a
`verify_issue` flow that resumes across turns) shows as one timeline.

**Enable it:** set three env vars (otherwise tracing is a complete no-op):

```bash
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=https://cloud.langfuse.com   # or your self-hosted URL
```

- **Quickest:** create a free project at https://cloud.langfuse.com and copy the keys.
- **Self-host:** run Langfuse's official stack (`git clone langfuse/langfuse && docker compose up`)
  and point `LANGFUSE_HOST` at it. (Langfuse is its own multi-container stack —
  web/worker/postgres/clickhouse/minio — so it lives in its own compose, not ours.)

All instrumentation goes through `agent_customer_support/observability/tracing.py`
(the only module that imports `langfuse`); it fails open — any tracing error is
logged and never breaks a request. Buffered traces are flushed on server shutdown.

## LLM layer note

The runtime LLM client is **vendored** in `agent_customer_support/llm/` (Anthropic +
OpenAI, selected by `AGENT_MODEL`); the agent does not import `enterprise_llm_service`.
Runtime deps (anthropic, openai) are declared in `pyproject.toml`.

`enterprise_llm_service` is only needed for the **offline KB indexing** script
(`scripts/index_kb.py`), installed as a local wheel (`--no-deps`). The dummy env vars
below exist solely so that script's imports resolve; they are not used by the agent.

Dummy env vars for the offline indexing script (already stubbed in conftest.py for tests):
- QDRANT_ENDPOINT, QDRANT_API_KEY, CELERY_BROKER_URL, CELERY_RESULT_BACKEND
- OPENAI_API_KEY, TOGETHERAI_API_KEY

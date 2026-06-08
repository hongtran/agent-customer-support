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

# 3. Install enterprise_llm_service LLM layer (manual step — not in Poetry registry)
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

## LLM layer note

`enterprise_llm_service` is installed as a local wheel (`--no-deps`). Runtime deps (anthropic, openai, tiktoken, google-genai, environs, tenacity) are declared in `pyproject.toml` and installed by `poetry install`. On a fresh checkout, you must re-run the `pip install --no-deps` step after `poetry install`.

Required dummy env vars for tests (already in conftest.py):
- QDRANT_ENDPOINT, QDRANT_API_KEY, CELERY_BROKER_URL, CELERY_RESULT_BACKEND
- OPENAI_API_KEY, TOGETHERAI_API_KEY

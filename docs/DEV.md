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

## Authentication

Every `/widget/*` and `/admin/*` route sits behind a bearer token. `POST /auth/login`
takes `{user_name, password}` and returns a JWT — **`user_name` is the `customer_id`**,
there is no separate username field. `/admin/*` additionally requires `role == "admin"`.

Set `JWT_SECRET` in `.env` first; the server refuses to start without it.

### Creating the first admin

There is no bootstrap code and no seed script — the first admin is a row you insert by
hand. It needs a bcrypt hash, not a plaintext password:

```bash
poetry run python -c \
  "from agent_customer_support.auth import hash_password; print(hash_password('your-password'))"
```

Then write the row (DynamoDB Local):

```bash
aws dynamodb put-item --endpoint-url http://localhost:8000 --table-name acs_customers \
  --item '{"customer_id":{"S":"admin"},"name":{"S":"Admin"},"role":{"S":"admin"},
           "password_hash":{"S":"<paste the hash>"},"enabled_applications":{"L":[]}}'
```

Every customer after that is created from the **Khách hàng** tab in the admin UI, or via
`POST /admin/customers`. Profiles that predate authentication have no `password_hash` and
cannot log in until an admin sets one.

## Running the agent

```bash
# Seed flows into DynamoDB
poetry run python scripts/import_flows.py seeds/flows

# Start the API server (port 8800)
make run

# Log in to get a token
TOKEN=$(curl -s -X POST http://localhost:8800/auth/login \
  -H "Content-Type: application/json" \
  -d '{"user_name":"ttp","password":"your-password"}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')

# Test the chat endpoint — note there is no customer_id in the body, it comes from the token
curl -X POST http://localhost:8800/widget/chat \
  -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" \
  -d '{"conversation_id":"cv1","message":"Làm sao xử lý PYC sự cố?"}'
```

## Admin conversation history

`/admin/customers/{id}` in the UI lists a customer's conversations and shows each one
turn by turn, with screenshots. The list reads the GSI `customer_id-updated_at-index` on
the conversations table (hash `customer_id`, range `updated_at`, both strings; projection
INCLUDE `title`, `created_at`, `turn_count`).

- **Local:** the API creates the index on start (`DYNAMODB_AUTO_CREATE_TABLES=true`),
  also on a table that already exists.
- **Prod:** with auto-create off, the index must be added by infrastructure first.

The index is sparse: conversations written before it existed have no `updated_at` and do
not appear until they get a new turn. Backfill them once (dry run without `--apply`):

```bash
poetry run python scripts/backfill_conversation_summary.py --apply
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

The agent emits a hierarchical trace per turn — root `turn` (type `chain`, one
conversation turn) → per-agent spans (type `agent`: `agent.triage`,
`agent.knowledge`, `agent.flow`, `agent.verification`, `agent.escalation`) → LLM
generations (`llm.<agent>`, with model + token usage), tool spans (type `tool`,
`tool.<name>`), and `rag.search` (type `retriever`). Traces are grouped by
`session_id = conversation_id`, so a whole multi-turn conversation (including a
`verify_issue` flow that resumes across turns) shows as one timeline.

**Every LLM generation names the agent that made it.** `tracing.agent_span` sets a
ContextVar that `tracing.generation` reads, so the call becomes `llm.knowledge`
rather than a bare `llm` and carries `metadata.agent`. That is what makes an
evaluator targetable at one agent: filter Langfuse on the observation name
`llm.knowledge`, or on `metadata.agent`. A second LLM call inside one agent adds a
step (`tracing.step`) — the knowledge query-rewrite is `llm.knowledge.contextualize`,
so it is not judged as if it were an answer. Note Langfuse `tags` are a *trace*-level
field and cannot separate observations; the type + name + metadata do that job.

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

## Self-hosted Qwen on Modal

`qwen/serve.py` runs a Qwen model behind vLLM's OpenAI-compatible API. The `PROFILE`
constant at the top picks the setup:

| Profile | Model | GPU | App model name |
|---|---|---|---|
| `l4-9b` (default) | `Qwen/Qwen3.5-9B` (BF16, ~19 GB) | L4 | `modal/qwen3.5-9b` |
| `h100-27b` | `Qwen/Qwen3.8-27B-FP8` (~28 GB) | H100 | `modal/qwen3.8-27b` |

`l4-9b` is a cheap **plumbing test** (routing, JSON answers, citations), not a quality
test: 9B with thinking turned off. **Do not use a T4**: vLLM hung compiling Qwen3.5's
Triton linear-attention kernels on Turing until the 20 min startup timeout.

It scales to zero. A request that arrives with no container running **waits** for the
cold start (several minutes) — it is a `@modal.web_server` Function, which holds the
request, not an `@app.server`, which answers 503. For production, use `h100-27b` and
add `min_containers=1` to `@app.function`.

```bash
uv tool install modal && modal setup                 # once: CLI + login
modal secret create qwen-vllm-api-key VLLM_API_KEY="$(openssl rand -hex 32)"
make modal-download                                  # once: 9B weights into the qwen-weights Volume
# for h100-27b instead: modal run qwen/download_model.py --repo-id Qwen/Qwen3.8-27B-FP8
make modal-deploy                                    # prints the server URL
```

Then in `.env`:

```bash
MODAL_LLM_BASE_URL=https://<workspace>--qwen-vllm-serve.modal.run/v1   # URL printed by deploy + /v1
MODAL_LLM_API_KEY=<same value as VLLM_API_KEY>
KNOWLEDGE_MODEL=modal/qwen3.5-9b        # or modal/qwen3.8-27b for the h100-27b profile
```

Smoke test (the first call may take minutes while the container starts; `-L` follows
Modal's 303 redirects during a long wait):

```bash
curl -L "$MODAL_LLM_BASE_URL/models" -H "Authorization: Bearer $MODAL_LLM_API_KEY"
```

**If the server fails to start**, watch `modal app logs qwen-vllm`:
- out of memory → lower `--gpu-memory-utilization` or `--max-num-seqs` (not `--max-model-len`:
  compose prompts are ~12K tokens plus a 4K output reserve, so below ~20K requests fail with 400)
- CUDA graph / compile error → add `--enforce-eager` (slower output)

Switch back to OpenRouter with `KNOWLEDGE_MODEL=openrouter/qwen/qwen3.8-27b`.

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

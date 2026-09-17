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
1. **Input guardrail** — cheap non-LLM checks (empty/oversized input) only
2. **Triage** — routes to `flow`, `escalate`, `knowledge`, or `out_of_scope` (clearly
   non-CenLab questions get the canonical refusal before any RAG/compose spend;
   this is the only scope gate — `KnowledgeAgent` deliberately carries no scope
   logic, so anything triage lets through gets a normal answer attempt)
3. **Knowledge** — RAG search + LLM answer; may detect a suspected bug
4. **Verification** — multi-turn evidence collection when a bug is suspected (state preserved in `session.pending = "verify_issue"`)
5. **Flow** — walks the user through a step/transition/outcome tree (e.g. account recovery)
6. **Escalation** — posts to Zalo webhook and returns a handoff reply
7. **Output guardrail** — grounding check on a reply that cited a passage; an
   unsupported answer is replaced and handed off (see Citations below)

Every agent receives a `TurnContext` (`agents/context.py`) and returns `AgentResult` (`models.py`). The `Agent` protocol (`agents/base.py`) is a structural interface — just `name: str` and `async def run(ctx) -> AgentResult`.

### LLM layer

`llm/__init__.py` exports `complete_with_tools` and `complete_text` — these are the only LLM call sites. Model routing is automatic, by model name: an `openrouter/` prefix goes to OpenRouter, a `modal/` prefix to the self-hosted vLLM server, names containing `"claude"` to Anthropic, and everything else to OpenAI. The two prefixed routes reuse the OpenAI provider with a different client; the prefix is stripped before the call. Per-agent model overrides are configured in `Settings` (`config.py`) and accessed via `settings.model_for("triage")` etc.

The OpenAI provider builds request params per model family (`llm/providers/openai_provider.py`): reasoning models (`gpt-5*`, `o1/o3/o4*`) get `max_completion_tokens` + `reasoning_effort` and **no** `temperature`; older models keep `max_tokens` + `temperature=0.5`. The facade resolves the reasoning profile once from `Settings` and passes it down, so providers stay pure functions of their arguments. Effort and the token ceiling are enforced by `ENVIRONMENT` alone — `dev` → `low`/4000, `prod` → `high`/8000 (`_REASONING_EFFORT_BY_ENV` in `config.py`); there is no per-key override by design.

**Self-hosted Qwen (`modal/…`).** `qwen/serve.py` runs vLLM on Modal with two profiles picked by the `PROFILE` constant: `l4-9b` (default; `Qwen/Qwen3.5-9B` on an L4 with thinking off — a cheap plumbing test, not a quality baseline; T4 hangs at startup) and `h100-27b` (`Qwen/Qwen3.8-27B-FP8` on an H100). Both scale to zero (the first call after 5 idle minutes waits for a cold start of minutes). It is a `@modal.web_server` Function, not `@app.server`: a Server answers 503 while no container is ready, which failed every cold-start turn; a web Function holds the request. `PROFILE` is a constant, not an env var, because Modal re-imports the file inside the container. It is guarded by vLLM's `--api-key`, not Modal proxy auth, so a plain OpenAI client works. Three details are load-bearing: `--reasoning-parser qwen3` keeps the thinking out of `message.content` — without it every `ComposedAnswer` parse fails; the env effort is translated through `_QWEN_REASONING_EFFORT` into `chat_template_kwargs`, because the Qwen3.8 chat template raises on anything but `low|medium|xhigh` (OpenAI's `high` included; the Qwen3.5 template ignores the kwarg); and `temperature=None` leaves sampling to the model's own `generation_config.json` instead of the 0.5 tuned for OpenAI models.

### RAG

`rag_client.py` reads Qdrant directly (no HTTP service in front of it). `_normalize_collection` maps the logical collection name from `Settings` to the physical `_v3` collection, mirroring enterprise-llm-service's `RagManager`. Per-document collapsing (`per_doc`) is skipped only for a **single**-application scope — one guide, where capping chunks per document would leave a single passage; a multi-application scope (always the shape of a widened retry) collapses so one guide can't crowd out the rest. Embeddings go through `rag/embeddings.py` (Google `gemini-embedding-001`).

Application scoping is a **hard, server-side Qdrant payload filter** (`_build_filter`) applied during the vector search, not a post-filter — so a rare application can't be squeezed out of the candidate set, and a scoped query never answers from another application's docs. One deliberate exception: a document with no `application` in its metadata (missing or `null`) is treated as **global** and stays visible to every customer, which is what keeps untagged Q&A records (`QARecord.application` is optional) reachable. Scoping is driven by `session.selected_applications`, with
`CustomerProfile.enabled_applications` as the ceiling (below).

**A scope that returns nothing is retried once, wider.** Because the filter is hard, a
user who picks the wrong module in the widget gets zero passages for a question the
corpus can answer one module over — and the composer then emits `[[no_answer]]`, so the
turn ends in a clarify-then-handoff instead of an answer. `RagClient.search_with_fallback`
retries that miss against a caller-supplied wider scope; `KnowledgeAgent` supplies
`CustomerProfile.enabled_applications`, **never** an unscoped search, so a customer is
never told about a module they did not buy. The trigger is zero passages, not a low
`top_confidence` — `score_threshold` has already applied, so an empty list is the only
unambiguous "this scope has nothing" signal, and passages that came back but don't
answer stay the composer's call as everywhere else. The retry is skipped whenever it
could not change the outcome (`_is_wider`), and the query is embedded **once** for both
attempts — hence the `_query` / `search` split, since a re-embed would be a second paid
call for a byte-identical string. The result carries `fallback_used` and
`applications_used`; on a widened hit `KnowledgeAgent` passes the foreign module's
display name into the compose prompt (`KNOWLEDGE_OTHER_APPLICATION_NOTE`, appended to
the **user** content so the `cache_control`'d system prefix is not invalidated) so the
reply tells the user which module the feature actually lives in.

**Application identifiers have two forms and the boundary matters.** Qdrant stores a **slug** (`lay_mau_quan_trac`); the rest of the stack uses **display names** (`Lấy mẫu - Quan trắc`) — `CustomerProfile.enabled_applications` holds names, `/widget/customers/{id}/applications` serves names, and the widget sends names back in `ChatRequest.applications`. `RagClient.search` translates via `applications.to_slugs` before it filters; filtering on a raw display name matches **nothing**. `applications.py` carries the canonical map, duplicated from enterprise-llm-service's `_APPLICATION_SLUGS` (`data_processing/extract_info_user_guide.py`) — keep them in sync. Note `seeds/flows` uses a third, kebab-case form; `to_slug` tolerates it. The `rag.search` span logs both `applications` and `applications_resolved` so a scoping miss is diagnosable from the trace.

**Payload indexes are mandatory, not an optimisation.** The Qdrant deployment runs strict mode with `unindexed_filtering_retrieve=False`, so filtering on a key with no keyword payload index fails with a 400 — there is no fallback scan. The product collection already has indexes on `metadata.application`, `metadata.doc_type`, and `metadata.job_role` (created by enterprise-llm-service's `rag/indexing.py`). For the Q&A collection, `rag/qa_indexer.ensure_collection` creates them — deliberately on every process start, not only at collection creation, because that collection predates the indexes.

### Authentication

Every `/widget/*` and `/admin/*` route requires a bearer JWT; `/admin/*` additionally
requires `role == "admin"`. `auth.py` is the only module that imports bcrypt or PyJWT.

**There is no separate user table.** `CustomerProfile` carries `password_hash` (bcrypt,
nullable — `None` means "cannot log in", which is what every pre-auth row looks like) and
`role`. **The login `user_name` IS the `customer_id`**, so login is a direct `get_item`
with no secondary index — the trade-off is that a username can't change without changing
the tenant id, and `POST /admin/customers` therefore refuses to overwrite an existing id
(`ConditionExpression`, 409) instead of letting `put_item` upsert a live tenant away.

**`ChatRequest` has no `customer_id`.** Identity comes from the token in
`get_current_customer` (`channels/deps.py`) and nowhere else. A client-supplied tenant id
was the original hole: it keys the conversation store and drives the Qdrant application
filter, so trusting it crossed the tenant boundary. For the same reason
`/widget/me/applications` has no path parameter, and `/widget/feedback` checks
conversation ownership before copying a transcript into the Q&A store.

`get_current_customer` re-reads the profile from `CustomerRegistry` rather than trusting
the token's claims beyond `sub` — the token is stateless and unrevocable, so that read is
what makes a deleted customer or a demoted admin take effect immediately instead of at
expiry. `role` is therefore always the stored one, never the minted one.

The first admin is inserted by hand (see `docs/DEV.md`); there is no bootstrap path and
no seed script by design.

### Storage

| Store | Backend | Purpose |
|---|---|---|
| `SessionStore` | Redis | Turn-to-turn state (`active_flow_id`, `pending`, TTL-based) |
| `ConversationStore` | DynamoDB | Full message history |
| `CustomerRegistry` | DynamoDB | Customer profiles & enabled modules |
| `FlowStore` | DynamoDB | Flow definitions (seeded via `scripts/import_flows.py`) |
| `RequestBacklog` | DynamoDB | Bug/feature/how-to records logged on escalation |
| `AttachmentStore` | S3 | Uploaded screenshot bytes; the turn keeps only the key |
| `UsageStore` | DynamoDB | Per-customer daily question counter (rate limit) |

**Daily question limit.** `CustomerProfile.daily_question_limit` is N (`None` = unlimited;
admins are never limited). `/widget/chat` calls `UsageStore.try_consume` after the 413 check
and before the turn; a refusal is a 429. The count lives in its own table, one item per
customer per day keyed `<customer_id>#<YYYY-MM-DD>` (day in `USAGE_TIMEZONE`, default
`Asia/Ho_Chi_Minh`). **The date in the key is the reset** — there is no midnight job that
could fail and leave customers blocked; old days expire via TTL on `expires_at`. It is not a
field on the customer row because the admin PATCH rewrites that whole item from an earlier
read and would roll the counter back. Check and increment are one conditional `UpdateItem`,
so concurrent requests cannot both take the last slot. A turn that fails later is not refunded.

**Attachments never carry bytes into DynamoDB.** A conversation is a single item that
`ConversationStore.append` rewrites on every turn, and DynamoDB caps items at 400 KB —
base64 inflates by 4/3, so a ~300 KB screenshot was enough to fail the write and take the
already-generated reply down with it. Three separate types enforce the split:
`Attachment` (inbound, has `data`, feeds the LLM) → `StoredAttachment` (persisted, has
`s3_key`, no bytes) → `AttachmentRef` (returned to the UI, presigned URL). Uploads are
size-checked at the widget boundary (413) before any S3 or LLM spend, and both the upload
and the presign in `Coordinator._finish` degrade on failure rather than raising — by that
point the reply is already paid for, so losing a screenshot beats losing the answer.

### Document images

Answers can show the screenshots and button glyphs from the source user guides. The guides
were converted from `.docx` with pandoc, so their chunks carry `![](media/image23.png)`
refs; `doc_images.py` is the whole text transform and `stores/doc_image_store.py` the S3
side. Images live at `<doc_images_prefix>/<application_slug>/imageNN.png`, uploaded by
`scripts/upload_doc_images.py`.

The pipeline: `rag.search` → rewrite refs to scoped markers → compose → validate/cap →
persist reply **with markers** → `Coordinator._finish` presigns for the response only.

**`media/imageNN.png` is unique only within one document** — every guide has its own
`image1.png` — so the key must be scoped by `metadata.application`. That is why the marker
is `[[img:<kind>:<slug>/<name>]]` and not just a filename, and why a chunk with no
`application` (the deliberate global-document case in `_build_filter`) has its refs dropped
rather than guessed at.

**Nothing is special-cased per document.** Whether a reply shows images is decided by what
is in the bucket at request time: `DocImageStore.names` is the whitelist, so a document
whose media has not been uploaded answers in plain text through the same code path as one
that has it. Uploading media is the entire integration step. A ref is *never* left
unresolved — it is rewritten or deleted, because a leaked `media/…` ref would be copied
through by the composer and render as a broken relative URL.

**The catalog, not the regex, is the hallucination guard.** `doc_images.select` checks each
composed marker against the same catalog the passages were rewritten against. Shape is not
enough: a model that invents `image999.png` under a real slug writes a perfectly
well-formed marker, and signing a URL for it would render a broken image. `select` also
dedupes and caps at `max_reply_images`, preferring `screen` over `icon`.

**The persisted turn keeps markers; only the response carries URLs.** A presigned URL
expires, so storing one would archive a dead link and feed ~500 characters of signature
into the transcript the LLM re-reads next turn — the same reasoning behind
`StoredAttachment` vs `AttachmentRef`. Resolution runs after the output guardrail, so the
guardrail judges prose. `kind` rides in the markdown alt text (`![screen](url)`) because
that is the only channel surviving into rendered markdown; the widget uses it to pick an
inline glyph vs a clickable preview thumbnail.

`kind` is derived from the ref's position in the source markdown — alone on a line means a
screenshot, sharing a line (a table cell) means a button glyph. That costs nothing, where
an object-size check would cost an S3 HEAD per image on the request path.

### Citations and grounding

An answer names its own sources. `_compose` returns a `ComposedAnswer`
(`llm/schemas.py`) — `{answer, cited}` — instead of free text, and `citations.py` is the
whole transform from that raw declaration to the source list the widget shows. This is a
**hybrid** on purpose: `answer` still carries every marker inline (`[[clarify]]`,
`[[no_answer]]`, `[[suspected_bug:…]]`, `[[img:…]]`) and `parse_markers` still reads them
out of it, so the heavily tuned compose prompt was appended to, never rewritten. Only the
citation list is promoted to a typed field.

**The catalog, not the shape, is the guard** — the same rule, for the same reason, as
`doc_images.select`. `citations.catalog` is built from this turn's `metas` *before*
compose, and `citations.select` keeps a declared id only if the catalog holds it. A model
that declares `"7"` when six passages came back writes a perfectly well-formed id, and a
citation the user cannot check is worse than no citation at all. Validation needs no
Qdrant round-trip: the passages and their metadata are already in memory.

**All three sources are declarable; only guides are displayed.** Product passages by
position (`"0"`, `"1"`), Q&A records by `"qa:<i>"`, and the always-on process block by the
pseudo-id **`quy_trinh_chung`**. A source row has to answer "where did this come from?"
with somewhere the reader can actually go, and the process block and the CS-verified Q&A
store are ours, not the customer's — naming them points at nothing they can open. So
`citations.select` keeps only `kind == "guide"`, and an answer resting solely on the other
two shows **no sources at all**, which is the honest outcome.

The other two stay *declarable* on purpose, and `select` is the only place they are
dropped. `passages_for` feeds the grounding judge from the same declarations, so removing
the Q&A ids would leave a mixed guide+CS answer judged against the guide alone and its
CS-derived claims flagged as unsupported. And taking the process id out of the prompt would
push the model to attribute a process claim to whichever passage is nearest — the
fabricated citation the whole mechanism exists to prevent.

**A citation names a section, never a file.** Each declaration carries the markdown heading
inside that passage whose content actually reached the answer (`CitedSource.section`), so a
row reads `Quy trình tổng thể – Vòng đời PYC · Yêu cầu thử nghiệm`. A chunk usually holds
several headings — often the guide's own `#` title above the `##` section it really covers
— so which one applies is not decidable from the chunk; only the answer knows.
`citations.sections` parses the candidates, and `_passages_block` lists them back to the
composer per passage (`(các mục trong đoạn này: … | …)`) so it **chooses from a closed set**
rather than guessing what looks like a title — left to guess, it picks the summary sentence
prepended to every chunk, which is not a heading and fails validation. The same function
serves both jobs on purpose: the list the model reads and the whitelist it is judged by
cannot drift apart. It is never shown to the *user*, though — listing every heading a chunk
contains would claim the answer used material it did not. A declared heading absent from
that passage is dropped and the citation survives without one, because pointing at the
wrong part of a guide is worse than pointing at the guide.

Both sides of that comparison run through `_clean_heading`, which strips the outline number,
markdown emphasis and a trailing colon — the guides write `##### **Import ký hiệu mẫu:**`
and the reader wants `Import ký hiệu mẫu`. Cleaning only the candidate would reject exactly
the well-behaved answers, since the prompt asks for the clean form.

Source **filenames never leave the server**. `Citation` has no field one could sit in and
`citations.py` holds none to put there — the constraint is structural, not a UI convention.
`label` falls back to the application display name for a heading-less chunk (the corpus
contains chunks that open mid-table), then to a fixed constant for an untagged global
document. `select` dedupes on `(label, application)` — what the user sees — so one guide
cited for two sections is two rows, while two heading-less chunks of one application are
one.

`AgentResult.citations` therefore means **"what the answer declared and we could verify"**,
not "everything retrieved" as it did before. The clarify and no-answer paths cite nothing:
those replies are canned text, not composed from a source.

The **output guardrail** (`agents/guardrail.py`) is the second half. It receives the reply
plus `AgentResult.cited_passages` — only the passages the answer stood on — and rules on
grounding alone; scope stays triage's job, and mixing the two is what made the previous
single moderation verdict hard to tune. **Empty `cited_passages` means no LLM call at
all**: every non-knowledge route and every clarify/process-only reply lands there, and
judging them would flag correct replies while spending a call on every turn. It still
fails OPEN. A flagged reply is replaced with `_FALLBACK_REPLY` and escalated, and its
citations are dropped — they vouched for text the user will never see.

`cited_passages` carries `exclude=True`: `Coordinator._traced` dumps every `AgentResult`
into a Langfuse span, and full passage text would bloat every trace.

Citations are **not** persisted on the `Turn` — reloading history shows answers without
their source lists.

### Flows

`models.py` defines `Flow → FlowStep → FlowTransition → FlowOutcome`. `FlowEngine` (`flows/engine.py`) is a pure stateless resolver — `FlowAgent` uses it to advance `session.current_step_id`. Flow JSON files are seeded from `seeds/flows/` via `scripts/import_flows.py`.

### Observability

All tracing goes through `observability/tracing.py` (the only file that imports `langfuse`). It is a no-op when `LANGFUSE_PUBLIC_KEY` is unset. Spans follow the hierarchy: `turn` (chain) → `agent.<name>` (agent) → `llm.<agent>[.<step>]` (generation) / `tool.<name>` (tool) / `rag.search` (retriever). The parenthesised names are Langfuse **observation types**, which the UI filters and renders separately. A generation is named after the agent that made it and carries `metadata.agent` (plus `metadata.step` for a second call inside one agent, e.g. `llm.knowledge.contextualize`) — that is the handle for pointing a Langfuse evaluator at one agent's calls. The agent name reaches the LLM facade through a ContextVar set by `tracing.agent_span`, so no agent signature carries it.

### Key env vars

See `.env-example`. The important runtime ones:
- `ENVIRONMENT` — `dev` (default) or `prod`; sets the enforced reasoning effort and output token ceiling. Prod deployments must inject it explicitly — the default is `dev`, i.e. `low` effort.
- `AGENT_MODEL` — default model (e.g. `gpt-5.4-mini`, `claude-sonnet-4-6`); per-agent overrides via `TRIAGE_MODEL`, `KNOWLEDGE_MODEL`, `KNOWLEDGE_CONTEXTUALIZE_MODEL`, `VERIFICATION_MODEL`, `FLOW_MODEL`, `GUARDRAIL_MODEL`
- `QDRANT_ENDPOINT` / `QDRANT_API_KEY` — Qdrant instance backing RAG; `GOOGLE_API_KEY` for the embedding model (`EMBEDDING_MODEL`, default `gemini-embedding-001`)
- `JWT_SECRET` — signs access tokens; **no default, the server refuses to start without it**.
  Anyone holding it can mint an admin token for any customer. `JWT_EXPIRE_MINUTES` (default 480)
  is the token lifetime; there is no refresh token, so a token is valid until it expires.
- `LANGFUSE_*` — optional tracing; leave blank to disable
- `DYNAMODB_ENDPOINT_URL` — set to `http://localhost:8000` for local dev
- `S3_ENDPOINT_URL` / `S3_BUCKET_ATTACHMENTS` — attachment storage; `http://localhost:4566` for LocalStack. `MAX_ATTACHMENT_BYTES` (default 5 MB) is the upload cap, `S3_PRESIGN_EXPIRY_SECONDS` (default 1h) the display-URL lifetime.

Note LocalStack is **not** part of `docker-compose.yml` — `make infra-up` starts only DynamoDB Local and Redis. Run LocalStack separately for the S3 path.

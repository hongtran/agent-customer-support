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

The `Coordinator` (`agents/coordinator.py`) runs the turn:
1. **Input guardrail** — cheap non-LLM checks (empty/oversized input) only
2. **Route** — `_route` is a driver loop over `routing.next_step` (see Routing below).
   The steps it can run:
   - **Triage** — routes to `knowledge`, `issue_verification`, `escalate`, or
     `out_of_scope` (clearly non-CenLab questions get the canonical refusal before any
     RAG/compose spend; this is the only scope gate — `KnowledgeAgent` deliberately
     carries no scope logic, so anything triage lets through gets a normal answer
     attempt)
   - **Knowledge** — RAG search + LLM answer, reporting `answer`, `clarify`,
     `no_answer` or `suspected_bug`
   - **Issue verification** — multi-turn slot filling when a bug is suspected (state
     in `session.pending = "verify_issue"`), reporting `need_more_info`, `user_error`
     or `bug_confirmed`. Reached two ways: a direct triage route when the user reports
     the software misbehaving, or Knowledge's `suspected_bug` status. Both go through
     `Coordinator._step_issue_verification`, so `pending_context` is armed identically
   - **File ticket** — a verified bug becomes a MantisBT ticket, a backlog row and a
     handoff, in that order (see Bug tickets below)
   - **Escalation** — posts to the Zalo webhook **and emails the CS team**, returns a
     handoff reply. Every handoff reply ends with a one-time ask for the user's phone
     and email (see Contact follow-up below)
3. **Output guardrail** — grounding check on a reply that cited a passage; an
   unsupported answer is repaired or handed off (see Citations below)

Every agent receives a `TurnContext` (`agents/context.py`) and returns `AgentResult` (`models.py`). The `Agent` protocol (`agents/base.py`) is a structural interface — just `name: str` and `async def run(ctx) -> AgentResult`.

### Routing

**The decision and the effects are separate files.** `agents/routing.py` holds
`next_step(RouteState) -> (Step, reason)`: a pure function over a small frozen
snapshot, with no I/O, no LLM call and no import of an agent or a store. Every rule is
therefore one assertion in `tests/agents/test_routing.py` with no mocks at all, where
reaching a branch of the old `_route` meant standing up five agents and nine stores.
`Coordinator._route` is the driver — it executes a step, updates the session, folds
the result back into the snapshot, and asks again until a step in `routing.TERMINAL`
ends the turn.

Because the driver owns the routing state, **no agent writes `session.pending` any
more**. `KnowledgeAgent` takes `allow_clarify` as an argument and reports a status; the
coordinator decides what that means for the session.

**Every rule carries a reason string**, and it is not decoration: it rides on the
step's Langfuse span as `metadata.handoff_reason`, and for the rules that end in a
handoff it is also the reason CS reads. The root `turn` span carries the whole `path`
(`["triage", "knowledge", "reply"]`), so a route can be read off a trace without
replaying it.

**The limits are the other reason this module exists.** There were none before: an
evidence collection that never completed kept `pending="verify_issue"` until the Redis
session expired, and nothing bounded a clarify loop. Each counter lives on
`SessionState` and has a rule that turns the cap into a handoff rather than a wait —
`MAX_CLARIFY` (2 questions, then escalate), `MAX_VERIFY_TURNS` (4 collection turns,
then file the ticket with whatever was collected) and `MAX_HOPS` (6) as the backstop
for the rules themselves. The longest legitimate path is four hops, so a run that
reaches six has found a cycle the rules did not anticipate.

`user_error_seen` is the one non-numeric guard: a `user_error` hands the turn back to
Knowledge, which can report `suspected_bug` again, and the two would ping-pong at a
model call each way. The second bounce escalates instead.

**Cancelling a flow is a regex, not a second triage pass** (`routing.wants_cancel`). It
runs on every turn with a flow open, and paying for an LLM call to read "thôi bỏ qua"
would be the most expensive way to read two words. It is conservative on purpose — the
bare verbs only count at the start of a message, because a false positive throws away
collected evidence while a miss only makes the user repeat themselves.

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
corpus can answer one module over — and the composer then reports `no_answer`, so the
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

### Bug tickets (MantisBT)

`IssueVerificationAgent` is **slot filling**: one structured call per turn
(`VerificationDecision` in `llm/schemas.py`) returning an outcome, a reply, and the whole
fixed set of facts a ticket needs (`BugSlots` in `models.py` — module, version, steps,
expected, actual, occurred_at). The bar used to be "at least ONE of an error message, a
screenshot or repro steps", read out of free text with an `[[evidence_ready]]` regex;
slots are what let Python see what is still missing and file a usable ticket even when
the collection is cut short.

**`slots.module` and `VerifyContext.application` are two different levels, and the
names are kept apart deliberately.** An *application* is what the customer bought: it
scopes the Qdrant filter, comes from the triage route or Knowledge's `suspected_bug`
status as a slug, and is what the ticket is filed under. A *module* is one level down
inside it — the menu, screen or page the user was on ("Danh sách phiếu yêu cầu"), which
only ever appears in the ticket body. Merging them into one field would scope a
retrieval to a screen name, which matches nothing in the corpus.

**Python is the memory, not the model.** Prior turns reach the model as plain text, so
a slot the model leaves empty means "nothing new this turn", never "forget that":
`BugSlots.merge` fills empty slots from what the session held and lets a non-empty
value win, so the user can still correct themselves. Whether a screenshot ever arrived
is tracked as `VerifyContext.has_image`, which is also why it is not a slot. The merged
context is written back to `pending_context` on **every** outcome — the old not-ready
branch returned no evidence at all, which is why no slot state could survive a turn.

**Earlier screenshots are shown again.** An image used to reach the model only on the
turn it was sent, so a screen name or an error dialog the model missed then was lost
for good. `_step_issue_verification` now reads the screenshots sent **since the start of
the conversation** (turn 0, not `since_turn` — the error screenshot often arrives with the
question Knowledge answered before the bug was suspected) back from S3
(`Coordinator._stored_images`, shared with the ticket's `_evidence_files`) and passes the
newest two as `ctx.evidence_images`; the agent attaches them to the slots note, leaving
the user's own message untouched. The ticket still takes only images since `since_turn`.
Every verification turn therefore pays an S3 read per earlier screenshot, and the
confirming turn reads the in-range ones twice (verifier, then ticket), which was left as
is rather than cached.

**A slot is asked at most once, in code.** A prompt alone did not hold this: in a live
conversation the model asked for the screen name three turns running after the user had
given it twice, because the old `module` wording ("inside the application, not the
application") made it treat "Quy chuẩn/Tiêu chuẩn" as a parent area. The model now also
returns `ask_for` — which slots its reply asks about — and `_guard` in
`issue_verification.py` enforces the rest, tracking `VerifyContext.asked_last` and
`ask_counts`:
- a slot we asked for last turn that the model still left empty is **filled with the
  user's message as written** (max 200 chars) — the message is the answer, the model
  just failed to read it. Known limit: an off-topic reply lands in the slot (a cancel
  phrase never gets here, `routing.wants_cancel` drops the flow first);
- a slot that is filled, or was asked once, is removed from `ask_for`, and if anything
  was removed the reply is rebuilt from `_SLOT_QUESTIONS`, so it cannot ask it anyway;
- while collecting: all required slots filled → `bug_confirmed`; everything the model
  asked was filtered and a required slot is still empty → one canned question for it,
  or `bug_confirmed` if each was already asked. A reply that asks for no slot at all (a
  screenshot) is left alone.
The guard trusts `ask_for`: a reply that asks for a slot without declaring it cannot be
seen, and the 4-turn cap is still the last stop.

**The docs are checked before any slot is asked.** On the first verification turn
(`VerifyContext.doc_checked`), `_doc_check` searches the guides with the same scope
rules as Knowledge (`search_with_fallback`, query = Knowledge's contextualized `query`
when the bug came from there, else the user's words) and makes one structured call
(`DocCheck`, traced as `llm.issue_verification.doc_check`) with `PROCESS_BLOCK` in the
system prefix and the passages numbered by `agents/passages.passages_block` — the same
sources and form the composer reads. `works_as_documented` with a cited id we actually
offered (a passage index or `quy_trinh_chung`) returns `user_error` at once; an invented
id, an empty explanation or a `None` parse falls through to slot filling, because
closing a real bug as a user error is the costly mistake. The model is called even with
no passages, since the process block alone can show the behavior is correct. Whatever
the verdict, `doc_expected` (what the docs say should happen, short text — never the
passages) is kept in `pending_context`: later turns read it in `VERIFICATION_SLOTS_NOTE`
("THEO TÀI LIỆU"), which is the only ground the slot-filling call has for a later
`user_error`, and `_with_slots` prints it in the ticket body.

**Three outcomes, and only one files a ticket.** `user_error` means the software
behaved correctly, so the turn goes back to Knowledge with the verifier's explanation in
`ctx.route_hint` — no MantisBT ticket, no backlog row, nobody paged. `need_more_info`
asks for at most two missing slots. `bug_confirmed` (or the collection cap, see Routing)
runs `Coordinator._step_file_ticket`: **report → MantisBT → backlog → Zalo**, in that
order. The report is a second structured call (`BugReport`: title, summary, repro steps;
traced as `llm.issue_verification.report`) over the same conversation; a `None` parse
falls back to `fallback_report`, which lifts a title from the user's original message,
because the ticket must still be filed once the evidence is in hand. A `None` parse on
the *decision* call, by contrast, always becomes `need_more_info` — a parse failure must
not be able to file a ticket. `_with_slots` folds the collected slots and the still-empty
required ones into the ticket body, so a report filed at the cap says so on its face
rather than reading like a complete one that happens to be thin.

`mantis.py` is the whole client and mirrors `escalation.py`: httpx, settings-driven, off
unless both `MANTIS_BASE_URL` and `MANTIS_API_TOKEN` are set. MantisBT's `summary` field
**is the title**, written as `[BUG][<customer name>] <title>` by `format_title` (VARCHAR
128: only the title part is cut, the prefix survives), `description` is
the body, the transcript goes in `additional_information`. The backlog row keeps the plain
title; it already carries the customer id. Screenshots are sent in a **second** call
(`POST /issues/{id}/files`) on purpose: a rejected upload (`max_file_size`) then costs
the screenshot, not the ticket. `create_issue` **never raises** — the reply is already
paid for, so a tracker outage costs the ticket only: the backlog row is still written
(with `mantis_issue_id = None`) and the Zalo note says the ticket must be filed by hand.
MantisBT goes before the backlog write so the row carries the ticket id in its single
`put_item`; there is no update path. Only the "verified bug" route files a ticket.

**The ticket is assigned to the customer's manager.** `CustomerProfile.mantis_handler_name`
(a MantisBT **username**, set in the admin form) becomes `"handler": {"name": …}` — a
name, not the numeric id, so CS can see in the form who manages the customer. MantisBT
rejects the whole issue when that user does not exist or cannot see the project, so a 4xx
on a create **with** a handler is retried once without it: a typo in the form costs the
assignment, not the ticket.

Evidence is **every image since the bug was suspected, and nothing before it**:
`_new_verify_context` stamps `since_turn` with the index the current
turn will get, and `_evidence_files` walks user turns from there (S3 keys, read back via
`AttachmentStore.get_bytes`) plus the current turn (still base64 in memory). A read that
fails drops that one file. Capped to the most recent `mantis_max_files`. The bytes exist
only for the duration of the turn — the `Attachment` / `StoredAttachment` split stands.

### Contact follow-up

The customer account is shared by a whole company, so after a handoff CS still needs to
know **who** to call back. The handoff itself is never delayed: ticket, backlog row, Zalo
and email all go out on the turn the escalation is decided, as before. Then, in one
place — `Coordinator._arm_contact_gate`, run in `handle_turn` whenever a result has
`escalated=True` — `ASK_CONTACT_REPLY` is appended to the reply and the session is put in
`pending = "collect_contact"` with `pending_context = {reason, backlog_id?,
mantis_issue_id?, mantis_issue_url?}`. Because it sits in `handle_turn`, all five paths
(triage, knowledge unresolved, verified bug, and the guardrail fallback) get it without knowing about it; the refs come from `AgentResult.escalation_refs`,
which only the verified-bug path fills. The gate must edit the session `_finish` saves
(`result.new_session` when set), or a path returning its own session would lose the flag.

The flag is **consumed on entry** to `_route`, before the verification/clarify branches,
and never re-armed by itself: the user is asked once. `contact.parse` (regex, no LLM:
a VN phone number or an email) decides what happens next. Found → `_attach_contact`
writes the `ContactInfo` onto the conversation record, the backlog row
(`RequestBacklog.set_contact`, a single-attribute update) and the MantisBT issue (as a
**note**, `MantisClient.add_note` — the ticket was filed before the contact existed), then
`Escalator.contact_update` tells CS again on both channels, and the reply is
`CONTACT_THANKS_REPLY`. Each step is best-effort and independent. Not found → nothing is
touched and the message is routed normally, because it is usually a new question.
`ContactInfo.raw` keeps the whole message so CS also sees "gọi sau 5h chiều". Nothing is
written to `CustomerProfile`.

`mailer.py` (`CSMailer`) is the email side: Resend's HTTP API (`POST /emails`) over
httpx, off unless `RESEND_API_KEY`, `CS_MAIL_FROM` and `CS_MAIL_TO` are set, never raises
(a rejected send is logged with Resend's own `message`, e.g. an unverified `from` domain). `Escalator.escalate` posts to
Zalo (still raises on an HTTP error) and then mails; the mail is sent even with no Zalo
webhook, and carries the full transcript where Zalo is capped at 3000 chars.

**Only three handoff reasons notify anyone.** `EscalationAgent` calls
`Escalator.escalate` only for `NOTIFY_REASONS` — `verified bug`, `knowledge unresolved`,
`clarify limit`. Every other reason (`user requested human`, `hop limit`, `bug loop`,
`ungrounded answer`) returns the same handoff reply and result shape (so the contact gate
still runs) but pages nobody. On a notified handoff, `CustomerProfile.manager_email` is
added as CC on top of `CS_MAIL_CC`; `CS_MAIL_TO` still gets every mail.

### Storage

| Store | Backend | Purpose |
|---|---|---|
| `SessionStore` | Redis | Turn-to-turn state (`pending`, `pending_context`, the routing counters, TTL-based) |
| `ConversationStore` | DynamoDB | Full message history |
| `CustomerRegistry` | DynamoDB | Customer profiles & enabled modules |
| `FlowStore` | DynamoDB | Flow definitions (seeded via `scripts/import_flows.py`) |
| `RequestBacklog` | DynamoDB | Bug/feature/how-to records logged on escalation |
| `AttachmentStore` | S3 | Uploaded screenshot bytes; the turn keeps only the key |
| `UsageStore` | DynamoDB | Per-customer daily question counter (rate limit) |
| `FeedbackStore` | DynamoDB | Like/dislike per assistant message, keyed `message_id` (latest vote wins, clear deletes) |

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
(`llm/schemas.py`) — `{answer, status, application, cited}` — instead of free text, and
`citations.py` is the whole transform from that raw declaration to the source list the
widget shows.

`status` (`answer | clarify | no_answer | suspected_bug`) used to be a marker written
inline in the prose and regexed back out; promoting it is what lets `routing.next_step`
read a validated value instead of re-parsing text. **The `[[img:…]]` markers stay
inline**, because they are positional — they mark *where* in the prose a screenshot
belongs — so there is nothing to promote. Two pieces of the old mechanism survive for
good reasons: `parse_markers` still serves the free-text retry in `_compose`, which has
no schema to fill, and `_scrub_markers` still strips control markers from `answer`
unconditionally, because a model told to use a field may write one into the prose out of
habit. Where the prose and the field disagree, the field wins. `_resolve_status` also
keeps the hedge rule: a full answer that *also* reports `no_answer` is a model hedging,
measured on the prose with image markers stripped first.

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
dropped. `passages_for` decides from the same declarations whether the grounding judge
runs at all, so removing the Q&A ids would skip the judge for an answer resting on a CS
record. And taking the process id out of the prompt would
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
plus `AgentResult.source_passages` — **every passage the turn retrieved** (guides and Q&A),
not only the cited ones, so a composer that cites the wrong chunk or forgets one it used
does not get correct claims flagged — and rules on grounding alone; scope stays triage's job, and mixing the two is what made the previous
single moderation verdict hard to tune. **Empty `source_passages` means no LLM call at
all**: `KnowledgeAgent` fills it only when `citations.passages_for` finds at least one
real cited passage, so every non-knowledge route and every clarify/process-only reply lands there, and
judging them would flag correct replies while spending a call on every turn. It still
fails OPEN.

**A failed verdict names each claim, and severity picks the repair.** The judge returns
`unsupported_claims` as `{span, replacement, severity, reason}` (`GroundingVerdict` in
`llm/schemas.py`): `span` is a verbatim substring of the reply, `replacement` is what it
should become so the sentence stays grammatical (empty = delete), and `severity` is
`minor` (extra but harmless, changes nothing the user does) or `major` (wrong or invented).

**`severity` is `minor | major`, but it does not gate the repair.** Every first failure
whose judge named at least one claim is repaired; severity only decides the rung, because
`apply_claims` edits minor claims only. Do not put a severity check back in front of the
ladder — that would send major-claim turns straight to escalation and skip the repair.

`Coordinator._repair_or_escalate` climbs a ladder, cheapest rung first:

1. **Python edit** — every claim is minor and `guardrail.apply_claims` can apply each
   replacement safely. **The replacement may only reuse the span's own words, in order**:
   `_safe_edit` checks that its words are a strict subsequence of the span's words
   (punctuation and case may change), so the judge can trim a sentence but never write a
   new claim into it. At most 12 words may be removed per claim; a pure delete is further
   held to a phrase (80 chars, not ending in `.!?`) because deleting a sentence can drop
   a step, where a replacement keeps it alive. Each span must be found exactly once,
   markers must survive, and the reply must keep some prose. All-or-nothing. The result
   is **not re-judged**: Python applied exactly the edit the judge named, so a second
   call would only confirm the judge's own list.
2. **LLM repair** — a major claim, or Python refused. One `KnowledgeAgent.repair`
   call (`llm.knowledge.repair`) with the same cited passages and the instruction "Xóa hoặc
   sửa các ý sau cho khớp với nguồn. Không thêm ý mới."; an image marker the original did
   not carry is dropped in code. The repaired reply **is** judged once more.
3. **Escalate** — a failure with no named claim (the repair call is skipped: it would be
   told to fix nothing), or a repair that still fails: the reply is replaced with
   `_FALLBACK_REPLY` and handed off, and its citations are dropped — they vouched for text
   the user will never see. A repaired reply keeps its citations, since a repair can only
   delete or reword, never add a source.

`eval/guardrail_eval.py` computes a `repair_path` column with the same rules
(`repair_path` must be kept in step with `_repair_or_escalate`), then runs that rung and
re-judges its output into the `repaired_*` columns.

`source_passages` carries `exclude=True`: `Coordinator._traced` dumps every `AgentResult`
into a Langfuse span, and full passage text would bloat every trace.

Citations are **not** persisted on the `Turn` — reloading history shows answers without
their source lists.

### Flows (parked — data layer only)

**There is no `FlowAgent`.** It was removed along with `SessionState.active_flow_id`, the
only thing that carried a flow between turns, so nothing in the live pipeline runs a flow
today. What remains is the data layer, kept so re-enabling means writing one agent again
rather than re-deriving the schema: `models.py` defines `Flow → FlowStep → FlowTransition
→ FlowOutcome`, `FlowEngine` (`flows/engine.py`) is a pure stateless resolver, `FlowStore`
holds the definitions, and `seeds/flows/` is seeded via `scripts/import_flows.py`.
`TurnContext.flow_store` is still wired but currently has no reader.

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
- `MANTIS_BASE_URL` / `MANTIS_API_TOKEN` / `MANTIS_PROJECT` — bug tickets; off unless the first two are set
- `RESEND_API_KEY` / `CS_MAIL_FROM` / `CS_MAIL_TO` — CS notification email via Resend; off unless all three are set. `CS_MAIL_CC` (comma-separated, optional) adds CC recipients. `CS_MAIL_FROM` must be on a domain verified in Resend
- `LANGFUSE_*` — optional tracing; leave blank to disable
- `DYNAMODB_ENDPOINT_URL` — set to `http://localhost:8000` for local dev
- `S3_ENDPOINT_URL` / `S3_BUCKET_ATTACHMENTS` — attachment storage; `http://localhost:4566` for LocalStack. `MAX_ATTACHMENT_BYTES` (default 5 MB) is the upload cap, `S3_PRESIGN_EXPIRY_SECONDS` (default 1h) the display-URL lifetime.

Note LocalStack is **not** part of `docker-compose.yml` — `make infra-up` starts only DynamoDB Local and Redis. Run LocalStack separately for the S3 path.

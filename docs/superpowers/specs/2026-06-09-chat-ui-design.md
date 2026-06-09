---
name: chat-ui-design
description: Next.js chat UI in ui/ subdirectory for testing AgentCore interactively in a browser
metadata:
  type: project
---

# Chat UI — Design Spec

**Date:** 2026-06-09  
**Status:** Approved

## Overview

A minimal Next.js (App Router + Tailwind) chat interface inside `ui/` for interactively testing AgentCore — the browser equivalent of `scripts/smoke_chat.py --interactive`. It calls the existing FastAPI `/widget/chat` endpoint; no new backend code is needed.

## Architecture

```
agent-customer-support/
├── agent_customer_support/   # existing FastAPI backend
├── ui/                       # NEW: Next.js app
│   ├── app/
│   │   ├── layout.tsx
│   │   └── page.tsx          # single-page chat UI
│   ├── components/
│   │   ├── ConfigBar.tsx     # customer_id + conversation_id inputs
│   │   ├── MessageList.tsx   # scrollable message thread
│   │   └── InputBar.tsx      # textarea + send button
│   ├── lib/
│   │   └── api.ts            # fetch wrapper for /widget/chat
│   ├── package.json
│   ├── next.config.ts
│   └── tailwind.config.ts
```

The Next.js dev server (`localhost:3000`) calls FastAPI (`localhost:8000`) directly. No proxy needed — CORS is handled by adding `localhost:3000` to FastAPI's allowed origins.

## Components

### ConfigBar
Top bar with two text inputs (pre-filled: `customer_id="ttp"`, `conversation_id="smoke-ui"`) and a **New conversation** button that resets `conversation_id` to a fresh `crypto.randomUUID()` value. Changes take effect on the next send.

### MessageList
Scrollable list of turns. User messages right-aligned (blue bubble), agent messages left-aligned (grey bubble). Auto-scrolls to bottom on new message. Shows a typing indicator (animated dots) while a request is in-flight.

### InputBar
Textarea at the bottom. **Enter** sends, **Shift+Enter** inserts newline. Send button disabled while loading or when input is empty.

## Data Flow

1. User types message and presses Enter / Send.
2. `api.ts` POSTs to `http://localhost:8000/widget/chat`:
   ```json
   { "customer_id": "ttp", "conversation_id": "smoke-ui", "message": "..." }
   ```
3. Response `{ reply, escalated, citations }` — only `reply` is rendered.
4. Messages stored in React state (`useState`). No persistence — refresh clears history.

## Error Handling

- Network/API errors show an inline red error message in the chat (e.g. "Error: could not reach server").
- Loading state prevents double-submit.

## Dev Setup

```bash
# Terminal 1 — FastAPI
poetry run uvicorn agent_customer_support.server:app --reload

# Terminal 2 — Next.js
cd ui && npm run dev
```

Open `http://localhost:3000`.

## Out of Scope

- Authentication
- Message persistence across page reloads
- Debug metadata display (tool calls, citations, escalated flag)
- Production deployment

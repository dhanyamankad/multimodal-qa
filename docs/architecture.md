# Architecture — Multimodal Q&A Pro

GenAI Summer of Code · Hackathon 2026 · Problem Statement 5

This document is the detailed companion to the overview in the root
`README.md`. It covers system design, data flow, module responsibilities,
and operational constraints — the things worth knowing before deploying or
extending the app.

## 1. System overview

The app is a **single-process FastAPI monolith**: one Uvicorn process serves
both the static frontend and the JSON/SSE API from the same origin, so there
is no CORS configuration and no separate frontend deployment.

```
Browser (static/ — HTML/CSS/JS, Tailwind, no build step)
        │  fetch() + EventSource, same-origin
        ▼
FastAPI app (main.py)
        │
        ├─ session state (in-memory, per session_id):
        │    _SESSION_IMAGES    — uploaded image paths this session
        │    _SESSION_HISTORY   — last N chat turns (for follow-up questions)
        │    _INGEST_PROGRESS   — live PDF-ingestion progress snapshots
        │
        ▼
agent/graph.py — one LangGraph create_react_agent (Groq openai/gpt-oss-120b)
        │  recursion_limit = 12, explicit routing rules in the system prompt
        │
        ├─ search_documents  → rag/retrieve.py → ChromaDB (session-scoped)
        ├─ search_web        → DuckDuckGo (ddgs)
        └─ describe_image    → agent/vision.py → Groq vision model
        │
        ▼
agent/synthesis.py — turns the agent run into:
        ├─ Chat Mode answer (plain text), or
        └─ Report Mode JSON (findings, conflicts, conclusion)
        │
        ▼
Response returned as JSON (/api/chat, /api/chat/report) or streamed as SSE
(/api/stream/{session_id}) with a live reasoning trace.
```

## 2. Request/response flow

1. Browser generates a `session_id` on page load and sends it with every
   request (chat, upload, stream).
2. `main.py` builds the message list for a turn: prior history
   (`_SESSION_HISTORY`) + the current query, annotated with image paths (if
   any were uploaded this session) and scope/cross-reference flags from the
   UI tab in use (`_build_full_messages`).
3. The LangGraph agent runs, calling tools per the routing rules in
   `agent/graph.py`'s system prompt (image-first, document-first, web
   conditional — see README "How routing works").
4. `agent/synthesis.py` turns the finished run into a Chat Mode string or a
   Report Mode JSON object.
5. The turn's query + answer are appended to `_SESSION_HISTORY` (trimmed to
   the most recent `_MAX_HISTORY_TURNS` turns) so follow-up questions have
   context.
6. For `/api/stream/{session_id}`, the same pipeline runs inside a single
   dedicated worker thread (via `loop.run_in_executor`) so the session
   contextvar (`agent/session_context.py`) survives for the whole run — a
   plain sync generator driven by Starlette can otherwise resume on a
   different thread mid-stream and silently break session isolation. The
   worker pushes serialized SSE events onto a `queue.Queue`; the async
   generator only drains that queue and never touches session state itself.

## 3. Module responsibilities

| Module | Responsibility |
|---|---|
| `main.py` | FastAPI app, routes, session/history/progress stores, SSE serialization |
| `agent/graph.py` | Builds the LangGraph `create_react_agent`, owns the routing system prompt |
| `agent/tools.py` | Implements `search_documents`, `search_web`, `describe_image`; each wrapped in `@safe_call` |
| `agent/safe_call.py` | Decorator: turns tool exceptions/timeouts into clean fallback strings instead of crashing the agent run |
| `agent/session_context.py` | contextvar-based per-thread session scoping (session id + allowed image paths) |
| `agent/vision.py` | Shared Groq vision-call helper: primary model + fallback model on failure |
| `agent/synthesis.py` | Converts a finished agent run into Chat Mode text or Report Mode JSON; strips stray citation-marker artifacts |
| `rag/ingest.py` | PDF → text (pypdf) → OCR fallback for image-only pages (PyMuPDF + Groq vision) → chunk (`RecursiveCharacterTextSplitter`, 500/100) → embed (`all-MiniLM-L6-v2`) → ChromaDB, session-tagged |
| `rag/retrieve.py` | Confidence-thresholded retrieval (similarity ≥ 0.08, calibrated empirically — see in-file note) so irrelevant chunks are reported `NOT_FOUND` rather than forced into an answer |
| `static/` | Plain HTML/CSS/JS UI (Tailwind via CDN, no build step) |

## 4. Data & state model

This is the part that matters most for choosing where to deploy.

- **Session state is process memory**, not a database: `_SESSION_IMAGES`,
  `_SESSION_HISTORY`, `_INGEST_PROGRESS` are plain Python dicts living
  inside the one running Uvicorn process. They disappear on restart, and
  they are **not** shared across multiple instances of the process.
- **ChromaDB is a local, on-disk `PersistentClient`** (`./chroma_db` by
  default). Every uploaded PDF's chunks live in a file on the same machine
  the app is running on. There's no external vector DB.
- **Uploaded files** (PDFs, images) are written to a local `uploads/`
  folder and referenced by path for the rest of the session.
- Together, this means the app needs **one long-lived process with a
  writable local disk that persists for at least the lifetime of a user's
  session** — not necessarily forever, but definitely longer than a single
  request.

## 5. External dependencies

- **Groq API** — both the reasoning LLM (`openai/gpt-oss-120b`) and the
  vision model(s) used for `describe_image` and OCR fallback. Requires
  `GROQ_API_KEY`.
- **DuckDuckGo search (`ddgs`)** — no API key required, used for
  `search_web`.
- **sentence-transformers (`all-MiniLM-L6-v2`)** — runs locally via
  `langchain-huggingface`, pulled in via PyTorch. This is the heaviest
  dependency in `requirements.txt` by far.

## 6. Known limitations (carried over from README, restated here for deploy planning)

- In-memory session store: a process restart clears every active
  session's history, images, and ingestion-progress state.
- Local ChromaDB: on any host with an ephemeral/reset-on-restart
  filesystem, uploaded documents stop being retrievable after a restart —
  this was already true on the original HF Spaces deployment and is true
  anywhere else with the same disk model.
- Single-process, single-instance design: there is currently no
  multi-instance session affinity or shared session store, so horizontal
  scaling (more than one running instance) would need sticky sessions or a
  shared store (e.g. Redis) added first.

## 7. Deployment implications

The app's Dockerfile builds a single container that runs Uvicorn on port
`7860` and expects to keep running (not invoked per-request), with a local
ChromaDB directory and an `uploads/` folder that need to survive for the
life of a session. The right hosting target is anything that runs **one
long-lived Docker container with a persistent-enough local disk** — see
`docs/deployment.md` for the specific platform decision and steps.
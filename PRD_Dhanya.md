# PRD — Dhanya's Scope

## Multimodal Q&A Pro — ByteSized Brains — Hackathon 2026 (PS5)

> **This is your working PRD, split from the master `PRD.md`.** It contains only what maps to your role, plus the shared context you need to not work blind. The full master PRD is the source of truth for anything not covered here — this file exists so you always know exactly what's yours, in what order, and what "done" looks like.
> 

> ⚠️ **MAINTENANCE RULE — READ THIS FIRST**
Every time Claude (or you) finishes a task, subtask, or checklist item from this PRD, **this file must be updated immediately** — check the box, add a one-line status note, and record any deviation from the plan (e.g. "swapped model X for Y because Z"). This is not optional bookkeeping — Vanshi is relying on this file being accurate to know what's ready to integrate against. **After every update, the full refreshed PRD content should be given back so you always have the latest copy in hand.** A stale PRD is worse than no PRD.
> 

---

## 1. Your Role (per master PRD Section 13)

**Dhanya — RAG pipeline, Frontend/UI architecture, deployment operations**

---

## 2. Your Owned Components

| Component | Path(s) | Master PRD reference |
| --- | --- | --- |
| PDF ingestion pipeline | `rag/ingest.py` | Section 4 (data flow), Section 6.1 |
| Retrieval logic (confidence threshold) | `rag/retrieve.py` | Section 6.1 |
| Custom UI (all 3 tabs) | `static/index.html`, `static/styles.css`, `static/app.js` | Section 5, Section 8 |
| Reasoning trace UI (collapsible accordion) | `static/app.js`, `static/styles.css` | Section 6.7, Section 4 |
| Report Mode UI rendering (findings/conflicts/conclusion) | `static/app.js` | Section 3.2 |
| Brand/design token application | all frontend files | Section 15 |
| HF Spaces Docker deployment | `Dockerfile`, `.dockerignore`, HF Space settings | Section 10 |
| Secrets setup on HF Spaces | HF Space Settings → Repository Secrets | Section 9 |
| README (setup + architecture + limitations) | `README.md`, `docs/architecture.md` | Section 8 |
| Cold-URL test (Test Scenario 5) | live HF Spaces URL | Section 11, item 5 |

---

## 3. Detailed Task Checklist

### 3.1 RAG Pipeline (`rag/`)

- [x]  `ingest.py`: PDF upload → extract text **with page numbers** — `pypdf`, 1-indexed pages
- [x]  `ingest.py`: chunk with `RecursiveCharacterTextSplitter`, `chunk_size=500`, `chunk_overlap=100`, chunked per-page (never across page boundaries, so every chunk has exactly one page number)
- [x]  `ingest.py`: embed with `sentence-transformers/all-MiniLM-L6-v2` via `HuggingFaceEmbeddings` — **confirmed fixed**: import is correctly `from langchain_huggingface import HuggingFaceEmbeddings`, and `langchain-huggingface` is present in `requirements.txt`. Re-verified directly against the actual file this update, not just claimed.
- [x]  `ingest.py`: store in ChromaDB **with page-level metadata** (filename + page number), via a `DocumentIngestor` class + module-level `get_ingestor()` singleton (shared client between ingest and retrieve, avoids file-lock contention)
- [x]  **NEW THIS UPDATE — critical interface bug found and fixed:** Vanshi's `main.py` imports a top-level `ingest_pdf(pdf_path, session_id=...) -> int` function that never existed in `ingest.py` (only the class method did) — every PDF upload was failing with a permanent 503. Added a real adapter function `ingest_pdf()` at module level that strips the `{session_id}_` filename prefix `main.py` adds before saving (so citations show clean filenames, not session-prefixed ones) and delegates to `DocumentIngestor.ingest_pdf()`, returning just the chunk count her endpoint expects.
- [x]  `retrieve.py`: implement similarity-confidence threshold — below threshold = "not found," never forced into an answer. Threshold empirically calibrated to `0.08` (see status log) after real testing showed the initial `0.35` rejected genuine answers
- [x]  Real interface: `retrieve(query, top_k, threshold) -> RetrievalResponse` with `.found` / `.chunks` / `.as_context_string()`, documented in the module docstring
- [x]  **NEW THIS UPDATE — critical interface bug found and fixed:** Vanshi's `agent/tools.py` imports `retrieve_chunks(query, threshold) -> list[dict]` (with `text`/`filename`/`page`/`score` keys) — a completely different name/signature/return type than the real `retrieve()`. Her own `try/except` around the import meant this failed silently and fell back to a stub always returning `[]`, so `search_documents` was permanently reporting `NOT_FOUND_IN_DOCUMENTS` even with real PDFs ingested. Added a thin `retrieve_chunks()` adapter in `retrieve.py` that wraps the real `retrieve()`/`RetrievalResponse` (no duplicated logic, single source of truth for the threshold).
    - **⚠️ OPEN — needs Vanshi to change one line on her side, not fixable from here:** her `agent/tools.py` currently hardcodes `CHUNK_CONFIDENCE_THRESHOLD = 0.35` and calls `retrieve_chunks(query, threshold=CHUNK_CONFIDENCE_THRESHOLD)`, which will override the new adapter's `0.08` default and silently reject real answers again. **Send her this exact note: delete her local `CHUNK_CONFIDENCE_THRESHOLD` constant and call `retrieve_chunks(query)` with no threshold arg (so it uses the real calibrated default), or update her constant to `0.08`.**
- [x]  **LOCKED DECISION: PDF-only ingestion, no xlsx/docx support.** Confirmed directly with Dhanya — the RAG pipeline stays scoped to `pypdf`parseable PDFs only, per the original locked stack. Merged UI's upload input only accepts `application/pdf`.

### 3.2 Custom UI — Structure & Tabs

- [x]  `index.html`: multi-tab layout — Tab 1 Hybrid Chat, Tab 2 Document Q&A, Tab 3 Image Studio — all three tabs merged into one real `static/index.html` with shared header + sidebar and working tab-switching JS.
- [x]  **NEW THIS UPDATE — full structural audit of `index.html` against `app.js`:** every one of the 34 DOM IDs `app.js` references (`report-title`, `docqa-ready-state`, `image-chat-thread`, etc.) exists exactly once, no duplicates or typos in either direction. Material Symbols icon font and Google Fonts (Space Grotesk / Inter / JetBrains Mono) both correctly linked in `<head>`. `app.js` correctly placed at end of `<body>`. File input `accept` attributes correctly scoped (`application/pdf` for docs, `image/*` for images). **No bugs found — file is clean as-is.**
- [x]  Chat Mode / Report Mode toggle on Hybrid Chat tab — real, working toggle in `app.js` (`setHybridMode()`)
- [x]  **NEW THIS UPDATE — critical bug found and fixed in `app.js`:** the Hybrid Chat send handler checked `if (currentHybridMode === "report")`, but `currentHybridMode` was never declared anywhere in the file — `setHybridMode()` toggled CSS classes but never tracked state in a variable. This threw a silent `ReferenceError` (swallowed as an unhandled promise rejection inside the async click handler) on **every single click of Send on the Hybrid Chat tab, in both modes** — the input cleared but no request ever fired, no visible error. Fixed by declaring `let currentHybridMode = "chat";` at top-level state and setting it inside `setHybridMode()`.
- [x]  PDF upload UI (Document Q&A tab) — real drag-and-drop + click-to-browse UI wired to `POST /api/upload/pdf`, with try/catch fallback to a clearly-labeled demo state when the endpoint isn't reachable yet
- [x]  Image upload UI (Image Studio tab) — real upload + preview wired to `POST /api/upload/image`, same graceful-fallback pattern
- [x]  Chat input + message thread UI (Hybrid Chat tab) — real `fetch()` calls to `POST /api/chat` / `/api/chat/report`, confirmed correct against Vanshi's real `main.py` response shapes (`data.answer`, nested `data.report`)
- [x]  No React/Node build step — vanilla JS only, per locked stack

### 3.2a Stitch UI Generation — Status

All three tabs generated, design-reviewed, exported, and merged into the real repo:

1. **Hybrid Chat** — Chat Mode + Report Mode (with-conflict and no-conflict variants) — ✅ DONE, exported, merged, **critical `currentHybridMode` bug found and fixed this update**.
2. **Document Q&A** — empty / populated / uploading states — ✅ DONE, exported, merged.
3. **Image Studio** — empty / populated states — ✅ DONE, exported, merged. **Known gap, not urgent:** the reasoning-trace block shown here is still hardcoded copy (`describe_image` + conditional `search_documents`) rather than driven by real `connectTraceStream()` events like Hybrid Chat's accordion is — flagged for a later pass, not blocking since the whole file is still gated by the Demo Mode badge.

**Bugs caught and fixed across this and prior review cycles:**

- Hybrid Chat Report Mode: a document was mislabeled "WEB" and didn't match its own citation — caught and fixed pre-export.
- Image Studio populated state: a single source was referred to by three different filenames across three UI elements — fixed, confirmed in final export.
- Document Q&A's reasoning trace was missing the Electric Cyan accent the other two tabs had — fixed via shared `.trace-container` class.
- `EV_Forecast_V3.xlsx` / `Competitor_Landscape_v2.docx` appearing as "indexed" in mockups — resolved via the locked PDF-only decision; real upload input only accepts `application/pdf`.
- **`.gitignore` encoding corruption, found and now actually fixed:** the file previously contained UTF-16 garbage on the `test.pdf` line (from a Windows tool), and a follow-up attempt to fix it went wrong a second way — a PowerShell heredoc command was pasted into the file as literal text instead of being executed as a command, so `.gitignore` briefly contained the command syntax itself rather than real ignore rules (meaning `.env`, `chroma_db/`, and `test.pdf` were all unprotected in the interim). Now genuinely fixed: clean UTF-8, LF endings, real gitignore patterns, verified via `git status`.
- **`rag/ingest.py` missing top-level `ingest_pdf()` — found and fixed this update** (see 3.1 above).
- **`rag/retrieve.py` missing `retrieve_chunks()` adapter — found and fixed this update** (see 3.1 above).
- **`app.js` `currentHybridMode` undefined-variable bug — found and fixed this update** (see 3.2 above).
- **No `.dockerignore` existed — found and fixed this update** (see 3.6 below): without one, `.env` (containing the real `GROQ_API_KEY`) would have been baked directly into Docker image layers on first build, a real secrets-leak risk surviving even a later `.env` removal from later commits.

### 3.3 Reasoning Trace UI

- [x]  `EventSource`/SSE client in `app.js` (`connectTraceStream()`), targeting `GET /api/stream/{session_id}?query=...` — confirmed against Vanshi's real `main.py`: endpoint requires `query` as a required query-string param, which `connectTraceStream()` correctly sends.
- [x]  Confirmed real SSE event shapes directly from her code: `{"type": "tool_call", "tool": ..., "args": {...}}`, `{"type": "tool_result", "tool": ..., "content": ...}`, `{"type": "ai_message", "node": ..., "content": ...}`, `{"type": "done"}` — `beginLiveTrace()`'s event handler matches all four.
- [x]  Collapsible accordion panel rendering tool name + order fired — implemented in Hybrid Chat's Chat Mode, driven by real `connectTraceStream()` events (not fabricated).
- [x]  Trace accent color = Electric Cyan `#22D3EE` — applied consistently across all three tabs via shared `.trace-container` CSS class.
- [ ]  **Image Studio's trace block is still hardcoded, not wired to real events** (see 3.2a) — real remaining work, not urgent.
- [ ]  Verify trace is driven by real events end-to-end — still blocked on confirming Vanshi's `agent.stream()` output live (backend exists now per her branch, but not yet tested end-to-end together).

### 3.4 Report Mode UI

- [x]  Render `findings[]`, `conflicts[]` (amber, only when populated), `conclusion` — implemented in `renderReport()`
- [x]  Confirmed empty `conflicts` renders cleanly — `.hidden` class toggle verified both in markup defaults and JS logic
- [x]  **Confirmed against Vanshi's real schema:** her `/api/chat/report` returns `{session_id, report: {findings, conflicts, conclusion}}` — no `title`/`subtitle` field, confirmed directly in both her `main.py` and `agent/synthesis.py`. `normalizeReport()` in `app.js` correctly unwraps `data.report` and generates `title`/`subtitle` client-side from the user's query rather than expecting the backend to supply them.

### 3.5 Design Tokens

| Pillar | Color | Where it applies |
| --- | --- | --- |
| Documents | Cobalt Blue `#2563EB` | `search_documents` related UI |
| Live Web | Teal `#0D9488` | `search_web` related UI |
| Vision | Coral `#F97066` | `describe_image` related UI |
| Conflict/alert | Amber `#F59E0B` | conflict-flag moments only |
| Reasoning trace | Electric Cyan `#22D3EE` | live trace panel only |
| Neutrals | Off-black `#0F172A` (dark), warm off-white `#F8FAFC` (light) | surfaces |
| Headings | Space Grotesk or Sora | — |
| Body | Inter | — |

**Status: all five accent tokens confirmed in active use** across the merged UI — no token silently unused.

### 3.6 Deployment

- [ ]  Push repo to GitHub (public) — **before pushing, run `git ls-files | grep .env` to confirm `.env` was never accidentally committed during the earlier `.gitignore` corruption window.**
- [x]  **NEW THIS UPDATE:** `.dockerignore` created — excludes `.env`, `.git`, `chroma_db/`, `venv/`, `__pycache__/`, `uploads/`, and other local-only files from the Docker build context. Without this, `.env` (with the real `GROQ_API_KEY`) would have been baked directly into image layers on first build — a real leak risk that would have survived even removing `.env` in a later commit, since old image layers persist. `Dockerfile` itself was already correct (base image, port 7860, `CMD` targeting `main:app`) — no changes needed there.
- [ ]  Create HF Space, SDK = **Docker**
- [ ]  Set `GROQ_API_KEY` under Space Settings → Repository Secrets (never hardcoded — check every commit before pushing)
- [ ]  Confirm build succeeds, container starts on port 7860
- [ ]  **Run the cold-URL test yourself**: fresh incognito window, live Space URL, full query end-to-end, zero prior setup (PS5 test scenario 5) — do this before telling Vanshi it's "done"

### 3.7 README & Docs

- [ ]  `README.md`: setup steps, architecture description, known limitations (mandatory per submission rules)
- [ ]  Embed live demo URL once deployed
- [ ]  Embed GitHub repo self-link
- [ ]  Embed the 9-scenario test checklist as evidence of rigor
- [ ]  `docs/architecture.md`: can mirror master PRD Section 4

---

## 4. Interfaces Confirmed From Vanshi (this update — pulled and reviewed her actual branch directly)

- **`/api/chat`**: `POST {session_id, query}` → `{session_id, answer}`. `app.js` correctly reads `data.answer`.
- **`/api/chat/report`**: `POST {session_id, query}` → `{session_id, report: {findings[], conflicts[], conclusion}}`. No `title`/`subtitle` — `app.js`'s `normalizeReport()` correctly generates these client-side.
- **`/api/upload/pdf`**: `POST multipart {session_id, file}` → `{session_id, filename, chunks_ingested}`. Requires the new `ingest_pdf()` adapter in `rag/ingest.py` (fixed this update) to stop 503-ing.
- **`/api/upload/image`**: `POST multipart {session_id, file}` → `{session_id, image_path}`. `app.js` correctly sends `session_id` as a form field.
- **`GET /api/stream/{session_id}?query=...`**: SSE events `tool_call` / `tool_result` / `ai_message` / `done`, confirmed exact field names against her code. `connectTraceStream()` matches.
- **`search_documents` interface (her `agent/tools.py`)**: expects `retrieve_chunks(query, threshold) -> list[dict]` with `text`/`filename`/`page`/`score` keys — now implemented as an adapter in `rag/retrieve.py` (fixed this update). **Still needs her to fix her hardcoded `0.35` threshold constant, see 3.1.**
- **`rag/ingest.py` interface (her `main.py`)**: expects a top-level `ingest_pdf(pdf_path, session_id=...) -> int` — now implemented as an adapter (fixed this update).

## 5. What You Must NOT Change Without Sign-Off

- The locked stack (no swapping vanilla JS for a framework, no swapping ChromaDB, etc.)
- The `.gitignore` / secrets handling — `.env` must never be committed, `GROQ_API_KEY` must never be hardcoded anywhere, including in Docker files or committed configs. `.gitignore` corruption (twice, see status log) is now fixed and verified; `.dockerignore` added this update to close the same class of risk on the Docker side.

---

## 6. Git Workflow

- Remote: `https://github.com/dhanyamankad/multimodal-qa.git`
- Branches (all lowercase): `main`, `dhanya`, `vanshi`
- Work on your own branch (`dhanya`), open a PR into `main` when a chunk of work is ready to integrate — don't push directly to `main`

## 8. Exactly What's Next (read this first if continuing in a new chat)

**Where things stand:** RAG pipeline is functionally complete and both critical interface adapters (`ingest_pdf()`, `retrieve_chunks()`) are now written and confirmed against Vanshi's actual current code (her branch was pulled and reviewed directly, not assumed). All three UI tabs are merged and working; the one bug that was silently breaking Hybrid Chat's Send button (`currentHybridMode`) is fixed. `.gitignore` is genuinely fixed this time (verified via `git status`, not just claimed). A `.dockerignore` now exists, closing a real secrets-leak risk that existed since the Dockerfile was first written.

**Next steps, in order:**

1. **Apply all 6 fixes to your actual local files** (you have the exact diffs from this session) and commit:
    
    ```
    git add rag/ingest.py rag/retrieve.py static/app.js .gitignore .dockerignoregit commit -m "Fix ingest_pdf/retrieve_chunks adapters, fix currentHybridMode bug, fix gitignore encoding, add dockerignore"git push origin dhanya
    ```
    
2. **Send Vanshi the one thing she needs to change on her side**: her `agent/tools.py` hardcodes `CHUNK_CONFIDENCE_THRESHOLD = 0.35`, which will override your calibrated `0.08` default the moment her import starts succeeding. She needs to either delete that constant and call `retrieve_chunks(query)` bare, or update it to `0.08`.
3. **Run an actual end-to-end local test** with both branches merged into a working tree: upload a real PDF, ask a question that should hit `search_documents`, confirm a real (non-demo, non-empty) answer comes back with correct citations. This is the first real integration test since both interface adapters landed — don't skip it.
4. **Then move to Section 3.6 (Deployment)** — nothing there has started except the `.dockerignore` just added: HF Space creation, secrets, Docker build, and the mandatory cold-URL test are all still open.
5. **Section 3.7 (README/docs)** — still fully open, can be done in parallel with deployment.

---

## 9. Status Log

*(Newest entry at top.)*

- `[This session: pulled Vanshi's actual vanshi branch and reviewed her real main.py / agent/tools.py / agent/synthesis.py directly (not assumed from an earlier diff). Confirmed two previously-suspected interface mismatches are real and fixed both: (1) rag/ingest.py had no top-level ingest_pdf() function at all despite main.py importing one — added an adapter that also strips the session_id filename prefix main.py adds, so citations stay clean; (2) rag/retrieve.py had no retrieve_chunks() adapter despite agent/tools.py importing one — added a thin wrapper around the real retrieve()/RetrievalResponse interface. Flagged an open item for Vanshi: her tools.py hardcodes threshold=0.35 which will override the real calibrated 0.08 default once her import stops silently failing.]`
- `[This session: found and fixed a critical app.js bug — currentHybridMode was referenced in the Hybrid Chat send handler but never declared anywhere, causing a silent ReferenceError on every Send click in both Chat and Report mode. Declared the variable and set it inside setHybridMode(). Also did a full structural audit of index.html against every DOM ID app.js references — no bugs found there, file confirmed clean.]`
- `[This session: found that an earlier attempted fix of the .gitignore encoding bug had gone wrong a second way — a PowerShell heredoc command got pasted into the file as literal text instead of being executed, so .gitignore contained command syntax instead of real ignore rules, leaving .env/chroma_db/test.pdf unprotected in the interim. Re-fixed properly this session and confirmed via git status.]`
- `[This session: found that no .dockerignore existed at all — COPY . . in the Dockerfile would have baked .env (containing the real GROQ_API_KEY) directly into Docker image layers on first build, a real leak risk surviving even a later .env removal since old layers persist. Created .dockerignore excluding .env, .git, chroma_db/, venv/, __pycache__/, uploads/, and other local-only paths. Dockerfile itself confirmed already correct — no changes needed there.]`
- `[LOCKED: PDF-only ingestion confirmed, no xlsx/docx support added.]`
- `[All 3 UI tabs generated in Stitch, exported, and merged into one real static/index.html + styles.css + app.js with working tab-switching, mode toggling, upload flows, and fetch()/EventSource calls to all 5 Section 7 endpoints with graceful demo-mode fallback.]`
- `[SIMILARITY_THRESHOLD recalibrated 0.35 -> 0.08 after real testing against a sample resume PDF.]`
- `[rag/ingest.py + rag/retrieve.py implemented — page-level PDF extraction (pypdf), RecursiveCharacterTextSplitter (500/100), all-MiniLM-L6-v2 embeddings, ChromaDB persistent storage, confidence-threshold retrieval.]`
- `[branches renamed to lowercase (dhanya, vanshi) and pushed to GitHub]`
- `[repo scaffolded — main branch committed, Dhanya/Vanshi branches created, .gitignore/.env.example/Dockerfile/requirements.txt/starter README populated]`
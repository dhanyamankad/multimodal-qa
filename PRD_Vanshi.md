# PRD — Vanshi's Scope
## Multimodal Q&A Pro — ByteSized Brains — Hackathon 2026 (PS5)

> **This is your working PRD, split from the master `PRD.md`.** It contains only what maps to your role, plus the shared context you need to not work blind. The full master PRD is the source of truth for anything not covered here — this file exists so you always know exactly what's yours, in what order, and what "done" looks like.

> ⚠️ **MAINTENANCE RULE — READ THIS FIRST**
> Every time Claude (or you) finishes a task, subtask, or checklist item from this PRD, **this file must be updated immediately** — check the box, add a one-line status note, and record any deviation from the plan (e.g. "swapped model X for Y because Z"). This is not optional bookkeeping — Dhanya is relying on this file being accurate to know what's ready to integrate against. **After every update, the full refreshed PRD content should be given back so you always have the latest copy in hand.** A stale PRD is worse than no PRD.

---

## 1. Your Role (per master PRD Section 13)
**Vanshi — Backend systems, agentic routing (LangGraph), LLM/synthesis logic**

`[OPEN: confirm this split is still accurate before build starts — adjust freely with Dhanya.]`

---

## 2. Your Owned Components

| Component | Path(s) | Master PRD reference |
|---|---|---|
| FastAPI backend + API contract | `main.py` | Section 7 |
| LangGraph agent setup | `agent/graph.py` | Section 4, Section 6 |
| Three tools (`search_documents`, `search_web`, `describe_image`) | `agent/tools.py` | Section 6.1–6.3 |
| `@safe_call` error-handling decorator | `agent/safe_call.py` | Section 6.4 |
| Routing system prompt rules | `agent/graph.py` / prompt config | Section 6.5 |
| Synthesis layer (Chat Mode vs Report Mode) | `agent/synthesis.py` | Section 6.6, Section 3.2 |
| Reasoning trace emission (real `agent.stream()`) | `agent/graph.py`, `main.py` SSE endpoint | Section 6.7 |
| Model deprecation check | pre-build action item | Section 5 critical action item |
| Test scenario scripting | `tests/test_scenarios.py` | Section 11 |

---

## 3. Detailed Task Checklist

### 3.0 Before You Write Any Model-Calling Code
- [x] Confirmed `openai/gpt-oss-120b` (reasoning) is currently live per `console.groq.com/docs` — no substitution needed
- [x] Confirmed `qwen/qwen3.6-27b` (vision) is currently live per `console.groq.com/docs/vision` — no substitution needed
- [x] Keep `meta-llama/llama-4-maverick-17b-128e-instruct` coded as a manual fallback vision model string (Section 5, Section 14) — **done, but see deviation note in status log**: this model is actually deprecated on Groq's side (announced 2026-02-20), and its official replacement is a text-only model, so there's currently no real second vision model to fall back to. Kept the exact string per explicit instruction; it will fail closed and `@safe_call` catches it, so this is really only demonstrating graceful degradation, not a working fallback. Flag for Dhanya/team decision before demo day.

### 3.1 FastAPI Backend (`main.py`)
- [x] `POST /api/chat` — Chat Mode conversational response
- [x] `POST /api/chat/report` — Report Mode structured JSON response (schema in Section 3.2 of master PRD)
- [x] `POST /api/upload/pdf` — hands off to Dhanya's `rag/ingest.py`; returns 503 with a clear message until she lands `ingest_pdf()` (currently an empty stub) — not a silent failure
- [x] `POST /api/upload/image` — stores image reference for session, available to `describe_image`
- [x] `GET /api/stream/{session_id}` — SSE stream of live reasoning trace, driven by real `agent.stream()` events
- [x] Serve static UI files from the same FastAPI app via `StaticFiles` mount — same-origin, no CORS config (deliberate, per Section 7); mounted last so it doesn't shadow `/api/*`

### 3.2 LangGraph Agent (`agent/graph.py`)
- [x] `create_react_agent` with all three tools attached, `recursion_limit=12` passed via `config=` at invoke/stream time (hard requirement, not a suggestion) — note: current langgraph (1.2.8) uses `prompt=` for the system prompt kwarg, not the older `state_modifier=`; confirmed via `inspect.signature` before writing this
- [x] System prompt encodes routing rules explicitly (Section 6.5) — image-first, doc-first, web-conditional-not-automatic, all spelled out as numbered rules rather than left implicit
- [x] Wire real `agent.stream()` output into the SSE endpoint — tool call events, in order, as they actually fire (Section 6.7); `_serialize_event` in `main.py` only ever forwards what's actually in the stream update, nothing fabricated

### 3.3 Tools (`agent/tools.py`)
- [x] `search_documents` — `@tool` decorated, calls `rag.retrieve.retrieve_chunks(query, threshold)`; threshold filtering is expected to happen on Dhanya's side. Her `rag/retrieve.py` is still an empty stub, so a clearly-labeled inline placeholder (returns no chunks → correct `NOT_FOUND_IN_DOCUMENTS`) stands in until she lands it — swap is a one-line change once her function exists
- [x] `search_web` — `DuckDuckGoSearchRun`, wrapped in `@safe_call(timeout=8.0)`; routing discipline (don't fire on doc-only questions) lives in the system prompt, verified this actually works via test scenarios 1 and 6
- [x] `describe_image` — calls Groq vision model in the same `@tool`-decorated function the agent calls directly, so it appears in `agent.stream()` like any other tool call — no separate code path

### 3.4 Error Handling (`agent/safe_call.py`)
- [x] `@safe_call`-style decorator wrapping every tool call path (all 3 tools)
- [x] On failure (timeout, API error, rate limit): clean fallback message returned to the agent; the fallback string flows through the tool's normal return value → `ToolMessage` → `agent.stream()`, so it shows up in the trace with zero extra logging machinery. Never propagates to an unhandled exception (broad `except Exception` is the safety net by design)
- [x] Deliberately tested, not assumed: ran a synthetic 5s-sleep tool through `safe_call(timeout=1.0)` — returned in ~1.0s with the graceful fallback string instead of hanging or raising. This is the exact mechanism PS5 test scenario 4 depends on.

### 3.5 Synthesis Layer (`agent/synthesis.py`)
- [ ] Chat Mode: natural conversational synthesis, sources implied not itemized
- [ ] Report Mode: structured JSON exactly per schema:
  ```json
  {
    "findings": [
      {"claim": "...", "source_type": "document", "source_detail": "filename.pdf, p.4"},
      {"claim": "...", "source_type": "web", "source_detail": "domain.com"}
    ],
    "conflicts": [
      {"topic": "...", "document_claim": "...", "web_claim": "...", "note": "..."}
    ],
    "conclusion": "..."
  }
  ```
- [x] `conflicts` populates **only** on genuine disagreement between doc and web on the same sub-topic — implemented as an explicit 2-step prompt (extract findings, then a dedicated sub-topic-by-sub-topic comparison step) rather than one implicit pass; prompt explicitly tells the model false positives are as bad as missed real conflicts (Section 16)
- [x] Every `findings` entry traceable to a real, specific source — `synthesize_report` is fed the actual `ToolMessage` contents pulled out of the finished agent run (`_extract_tool_outputs`), not re-derived from the chat-mode answer, so there's nothing to hallucinate a source for

Chat Mode (`synthesize_chat`) and Report Mode (`synthesize_report`) both implemented in `agent/synthesis.py`.

### 3.6 Test Scenarios (`tests/test_scenarios.py`)
Script all 9 — don't rely on manual-only testing before final QA:
- [x] 1. Pure doc question → only `search_documents` fires
- [x] 2. Image + doc cross-reference → trace shows `describe_image` → `search_documents` in order
- [x] 3. Current-info question, no relevant docs → clean fallback to `search_web` alone
- [x] 4. Simulated web search timeout → reported gracefully, no crash (also verified standalone, see 3.4)
- [x] 5. Local cold-start full query equivalent — asserts a real end-to-end answer is produced
- [x] 6. Doc-only question → no unnecessary web search triggered
- [x] 7. No answer in docs, answerable on web → clean fallback, no forced doc answer
- [x] 8. Outdated doc fact vs. current web fact → `conflicts` correctly populated with both claims
- [x] 9. Multi-part question needing both sources → every claim attributed to correct specific source

All 9 are scripted with deterministic monkeypatched tool internals (fake retrieved chunks, fake DuckDuckGo results) so they don't depend on live document/web content — but each one still drives the *real* `create_react_agent` graph against the live Groq reasoning model, so the actual routing decisions are genuinely under test, not mocked away. **Run live and passing**: `python -m pytest tests/test_scenarios.py -v -s` → `9 passed in 64.58s`. Two real bugs were found and fixed via this run (see status log) — the suite isn't just syntactically sound, it caught actual problems and confirmed the fixes.

---

## 4. Interfaces You Need From Dhanya
- Finalized shape of chunks/metadata coming out of `rag/retrieve.py` (so `search_documents` tool can consume it directly)
- Confirmation of the static file mount path/structure so `main.py`'s `StaticFiles` config matches what she builds

## 5. What You Must NOT Change Without Sign-Off
- `recursion_limit=12` on the agent (explicit PS5 requirement)
- The locked stack in master PRD Section 5 (LangGraph `create_react_agent`, Groq models, ChromaDB, DuckDuckGo)
- The API contract in master PRD Section 7 — if you need to diverge, flag it to Dhanya first since her frontend calls these exact endpoints

---

## 6. Git Workflow
- Remote: `https://github.com/dhanyamankad/multimodal-qa.git`
- Branches (all lowercase): `main`, `dhanya`, `vanshi`
- Work on your own branch (`vanshi`), open a PR into `main` when a chunk of work is ready to integrate — don't push directly to `main`
- Clone the repo and set your git identity locally before your first commit so authorship shows correctly on GitHub:
  ```
  git clone https://github.com/dhanyamankad/multimodal-qa.git
  cd multimodal-qa
  git checkout vanshi
  git config --global user.name "<your GitHub username>"
  git config --global user.email "<your GitHub email>"
  ```

## 7. Status Log
*(Newest entry at top. One line per completed item — what was done, any deviation, date/time if useful.)*

- `[2026-07-07, later same day] ALL 9 TEST SCENARIOS PASSING LIVE against real Groq calls (python -m pytest tests/test_scenarios.py -v -s -> 9 passed in 64.58s). This supersedes the earlier "scripted but not yet run live" note below — they've now actually been run, not just written. Two real bugs found and fixed in the process, both now verified fixed by the passing suite:`
- `[BUG FIXED] requirements.txt had duckduckgo-search, but langchain's DuckDuckGoSearchRun now imports the renamed "ddgs" package -> search_web failed 100% of the time with "Could not import ddgs". Swapped the requirements.txt entry to "ddgs". Also added missing load_dotenv() to tests/test_scenarios.py (main.py had it, the test file didn't, so GROQ_API_KEY wasn't visible when running pytest standalone).`
- `[BUG FIXED] Routing prompt (agent/graph.py SYSTEM_PROMPT) was too strict about "don't call search_web if search_documents already answered" -> caused test 8 and test 9 to fail because the agent stopped after search_documents even when the question EXPLICITLY asked to check/compare both sources. Added rule 3b: explicit multi-source requests override the "don't call both reflexively" rule. Also tightened agent/synthesis.py's conflict-check step -> it was initially treating "document says X (as of 2022), web says Y (as of now)" as "different time periods, not a real conflict," which is wrong for PS5 test scenario 8's whole premise (outdated doc vs current web fact). Made explicit that outdated-vs-current is a genuine conflict, not a borderline case.`
- `[known open item, needs a team decision before demo] meta-llama/llama-4-maverick-17b-128e-instruct (vision fallback) is deprecated on Groq's side and has no real replacement — see earlier deviation note below. It fails closed reliably; team should decide whether to keep as a graceful-degradation demo or remove the fallback branch.`
- `[minor, not urgent] Two library deprecation warnings surfaced during the live test run: create_react_agent is moving to langchain.agents in a future LangGraph version, and langchain-community is being sunset upstream. Nothing broken today — worth a line in README's "Known limitations" before final submission.`
- `[2026-07-07] Full first pass of Vanshi's scope built and committed locally on vanshi branch (agent/safe_call.py, agent/tools.py, agent/graph.py, agent/synthesis.py, main.py, tests/test_scenarios.py). All files syntax-checked and import cleanly; safe_call's timeout/fallback behavior verified directly (returns in ~timeout, not hanging, no unhandled exception). Full test suite is scripted but NOT yet run live — no GROQ_API_KEY / no outbound network to api.groq.com in this build sandbox, so scenarios are written with pytest.mark.skipif and need a real run before final QA.]`
- `[DEVIATION — flag before demo]` The manual vision fallback string `meta-llama/llama-4-maverick-17b-128e-instruct` (kept per explicit Section 3.0/14 instruction) is confirmed deprecated by Groq as of 2026-02-20; Groq's own recommended replacement for it is a text-only model, so there is currently no real second vision model on Groq to fall back to. Kept the string as instructed — it fails closed and `@safe_call` catches it cleanly — but this means `describe_image` effectively has no working fallback today, only the primary `qwen/qwen3.6-27b`. Worth a team decision: drop the fallback branch entirely, or accept it's fail-closed-only for now.
- `[INTERFACE STILL OPEN]` `rag/retrieve.py` and `rag/ingest.py` are still empty stubs (Dhanya's scope, untouched by this pass). `agent/tools.py`'s `search_documents` expects `rag.retrieve.retrieve_chunks(query: str, threshold: float) -> list[dict]` with each dict shaped `{"text": str, "filename": str, "page": int, "score": float}`, already filtered to `score >= threshold` on her side. Until that lands, a clearly-labeled inline placeholder returns no chunks (correctly triggers `NOT_FOUND_IN_DOCUMENTS` → web fallback, so the agent stays usable end-to-end). `main.py`'s `/api/upload/pdf` similarly expects `rag.ingest.ingest_pdf(path, session_id) -> int` (chunk count) and returns a clear 503 rather than a fake success until that exists.
- `[branches renamed to lowercase (dhanya, vanshi) and pushed to https://github.com/dhanyamankad/multimodal-qa — main, dhanya, vanshi all live on GitHub]`
- `[repo scaffolded — main branch committed, Dhanya/Vanshi branches created, agent/ and rag/ file stubs created (graph.py, tools.py, safe_call.py, synthesis.py, ingest.py, retrieve.py), main.py stub created]`

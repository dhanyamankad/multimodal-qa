---
title: Multimodal Q&A Pro
emoji: 🧠
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# Multimodal Q&A Pro — ByteSized Brains

GenAI Summer of Code · Hackathon 2026 · Problem Statement 5 (Hard)

## Team
- Dhanya — RAG pipeline, Frontend/UI architecture, deployment operations
- Vanshi — Backend systems, agentic routing (LangGraph), LLM/synthesis logic

## What this is
A single LangGraph `create_react_agent` with three tools (`search_documents`,
`search_web`, `describe_image`) that routes intelligently across documents,
live web search, and vision input, then synthesizes one answer — with an
optional "Report Mode" toggle for structured, source-attributed investigation
reports.

## Live Demo
`[ADD HF SPACES URL HERE BEFORE SUBMISSION]`

## Setup (local development)
1. `git clone <this-repo-url>`
2. `cd bytesized-brains-multimodal-qa`
3. `python -m venv .venv && source .venv/bin/activate`
4. `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and add your real `GROQ_API_KEY`
6. `uvicorn main:app --reload --port 7860`
7. Open `http://localhost:7860`

## Architecture
See `docs/architecture.md` for the full diagram and data flow. In short:

Custom UI → FastAPI (Uvicorn) → LangGraph `create_react_agent`
(recursion_limit=12) → three tools (ChromaDB / DuckDuckGo / Groq Vision) →
synthesis layer (Chat Mode or Report Mode) → response + live reasoning trace
streamed back over SSE.

## Test scenarios covered
**PS5 (organizer-mandated):**
1. Pure document question → only `search_documents` fires
2. Image + doc cross-reference → `describe_image` → `search_documents` in order
3. Current-info question, no relevant docs → clean fallback to `search_web`
4. Simulated web search timeout → UI reports failure gracefully, no crash
5. Cold live HF Spaces URL → full query works, zero local setup

**PS4-style (self-imposed, via Report Mode):**
6. Doc-only question → no unnecessary web search
7. No answer in docs, answerable on web → clean fallback
8. Outdated doc fact vs. current web fact → `conflicts` correctly populated
9. Multi-part question needing both sources → every claim attributed correctly

## Known limitations
`[Fill in before submission — e.g. rate limits on free-tier Groq, ChromaDB
is local/in-memory so it resets on container restart, vision model is
preview-tier and may be swapped, etc.]`

## Secrets
`GROQ_API_KEY` is read via `os.environ.get("GROQ_API_KEY")`. Locally it comes
from `.env` (gitignored). On HF Spaces it's set under
Settings → Repository Secrets — never hardcoded.

"""
The three tools available to the agent: search_documents, search_web, describe_image.

Routing discipline (who calls what, when) lives in the system prompt in
agent/graph.py, NOT here — this file only implements what each tool actually
does once called. See PRD Section 6.5 for the routing rules themselves.

Every tool is wrapped in @safe_call so a failure becomes a clean fallback
string instead of an unhandled exception (Section 6.4).
"""

from __future__ import annotations

import base64
import os

from langchain_core.tools import tool

from agent.safe_call import safe_call
from agent.session_context import get_session_id, is_allowed_image

try:
    from groq import Groq
except ImportError:  # pragma: no cover - dependency should always be installed per requirements.txt
    Groq = None  # type: ignore

# --- Interface with Dhanya's rag/retrieve.py -------------------------------
# EXPECTED CONTRACT (flagged in PRD Section 4 as an open interface item):
#   retrieve_chunks(query: str, threshold: float) -> list[dict]
#   each dict: {"text": str, "filename": str, "page": int, "score": float}
#   only chunks with score >= threshold are returned (the "above-threshold"
#   filtering happens on Dhanya's side in rag/retrieve.py, not here).
# rag/retrieve.py is currently an empty stub (owned by Dhanya). Importing it
# will fail until she fills it in, so we degrade to a clearly-labeled inline
# placeholder for local dev/testing rather than crashing at import time.
try:
    from rag.retrieve import retrieve_chunks, SIMILARITY_THRESHOLD as CHUNK_CONFIDENCE_THRESHOLD  # type: ignore
except Exception:  # noqa: BLE001
    # CHUNK_CONFIDENCE_THRESHOLD fallback below is NOT a guess — it mirrors
    # Dhanya's calibrated SIMILARITY_THRESHOLD in rag/retrieve.py (0.08,
    # picked from real testing: genuinely relevant chunks scored ~0.14-0.17,
    # irrelevant ones scored negative). The old 0.35 here would reject every
    # real answer even once retrieve.py exists, so this must stay in sync —
    # only used at all if the import above fails (rag/retrieve.py missing).
    CHUNK_CONFIDENCE_THRESHOLD = 0.08

    def retrieve_chunks(query: str, threshold: float = CHUNK_CONFIDENCE_THRESHOLD, session_id=None):
        """TEMPORARY PLACEHOLDER — replace by importing Dhanya's real
        rag/retrieve.py once it's implemented. Returns no chunks, which makes
        search_documents correctly report NOT_FOUND so the router falls back
        to search_web, keeping the agent usable end-to-end before RAG lands."""
        return []

# --- Vision models (see PRD 3.0 / Section 5 / Section 14) ------------------
# Confirmed live against console.groq.com/docs as of this build (2026-07-07):
VISION_MODEL_PRIMARY = "qwen/qwen3.6-27b"
# NOTE (status-log-worthy deviation, flagged not silently applied): Groq
# deprecated meta-llama/llama-4-maverick-17b-128e-instruct on 2026-02-20 in
# favor of openai/gpt-oss-120b (a text-only model, not a vision replacement).
# PRD Section 3.0/14 explicitly says to keep this string coded as the manual
# fallback regardless, so it stays below. In practice this means the fallback
# branch will reliably fail closed and get caught by @safe_call — which is a
# fine demonstration of graceful degradation, just not a working fallback.
# If a second real vision model is ever needed, there currently isn't a clean
# one on Groq (llama-4-scout was ALSO deprecated 2026-06-17, recommended
# replacement is qwen/qwen3.6-27b itself, i.e. the primary, not a distinct
# fallback).
VISION_MODEL_FALLBACK = "meta-llama/llama-4-maverick-17b-128e-instruct"

_groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY")) if Groq else None


@tool
@safe_call(fallback_message="Document search is temporarily unavailable.")
def search_documents(query: str) -> str:
    """Search the user's uploaded documents for information relevant to the query.

    Use this FIRST whenever a question could plausibly be answered from
    uploaded PDFs, before ever considering search_web. Returns only
    above-confidence-threshold chunks with page-level source metadata for
    citation, or an explicit NOT_FOUND signal if nothing cleared the bar.
    """
    # session_id comes from the contextvar set by main.py for this request,
    # NEVER from the LLM — see agent/session_context.py for why. This is
    # what makes each session's document Q&A strictly scoped to only the
    # files that session itself uploaded.
    chunks = retrieve_chunks(
        query, threshold=CHUNK_CONFIDENCE_THRESHOLD, session_id=get_session_id()
    )
    if not chunks:
        return (
            "NOT_FOUND_IN_DOCUMENTS: no above-threshold chunks matched this "
            "query in the uploaded documents. It is now appropriate to try "
            "search_web if the question needs an answer from somewhere."
        )
    formatted = [
        f"[{c['filename']}, p.{c['page']}] {c['text']}" for c in chunks
    ]
    return "\n\n".join(formatted)


@tool
@safe_call(fallback_message="Web search is temporarily unavailable.", timeout=8.0)
def search_web(query: str) -> str:
    """Search the live web for current information.

    Use this ONLY if search_documents returned NOT_FOUND_IN_DOCUMENTS, OR the
    question is explicitly about current/live information that uploaded
    documents cannot possibly contain (today's news, current prices, live
    scores, "as of today", etc). Do NOT call this reflexively alongside
    search_documents just because both tools are available — that is PS5's
    own test case 1 and the single most common failure mode.
    """
    from langchain_community.tools import DuckDuckGoSearchRun

    search = DuckDuckGoSearchRun()
    result = search.run(query)
    if not result:
        return "NOT_FOUND_ON_WEB: web search returned no usable results for this query."
    return result


def _vision_call(model_name: str, image_b64: str, question: str) -> str:
    response = _groq_client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
        max_tokens=800,
    )
    return response.choices[0].message.content


@tool
@safe_call(fallback_message="Image analysis is temporarily unavailable.")
def describe_image(image_path: str, question: str = "Describe this image in detail.") -> str:
    """Analyze an image uploaded in the current turn and answer a question about it.

    Use this whenever an image was uploaded in the current turn — it should
    fire before (or alongside, when cross-referencing) any document/web search
    needed to complete the answer, per PRD Section 6.5 routing rule 1.

    If multiple images were uploaded in this session, call this once per
    image path that's relevant to the question (the available paths are
    listed in the injected session context in the user message).
    """
    if _groq_client is None:
        raise RuntimeError("Groq client not configured — check GROQ_API_KEY")

    # image_path is an LLM-supplied argument. Verify it's actually one of
    # this session's own uploads before ever touching the filesystem, so a
    # hallucinated or injected path can't read another session's image (or
    # any other file on disk).
    if not is_allowed_image(image_path):
        return (
            "This image path is not part of the current session's uploads, "
            "so it was not opened. Only reference images uploaded in this session."
        )

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    try:
        return _vision_call(VISION_MODEL_PRIMARY, image_b64, question)
    except Exception:
        # Manual fallback per PRD Section 3.0/14. See VISION_MODEL_FALLBACK
        # comment above re: this model being deprecated on Groq's side —
        # if this also raises, the outer @safe_call catches it cleanly.
        return _vision_call(VISION_MODEL_FALLBACK, image_b64, question)
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

from langchain_core.tools import tool

from agent.safe_call import safe_call
from agent.session_context import get_session_id, is_allowed_image
from agent.vision import vision_call

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
        f"[{c['filename']}, p.{c['page']}{' (OCR transcription)' if c.get('ocr') else ''}] {c['text']}"
        for c in chunks
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

    # vision_call already tries VISION_MODEL_PRIMARY then VISION_MODEL_FALLBACK
    # internally (see agent/vision.py); if both fail it raises, and the outer
    # @safe_call turns that into a clean fallback string.
    return vision_call(image_b64, question)
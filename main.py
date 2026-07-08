"""
FastAPI backend + API contract (PRD Section 3.1 / master PRD Section 7).

Serves the static UI from the same origin as the API (deliberate — no CORS
config needed, per Section 7) and exposes the chat/report/upload/stream
endpoints the frontend calls against.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Without this, the root logger defaults to WARNING and INFO-level progress
# logs (e.g. rag/ingest.py's per-page OCR progress) are silently dropped --
# which is exactly what made a slow-but-working upload look identical to a
# hung one in the terminal. uvicorn configures its own loggers but not the
# app's; this line is what makes ours actually show up.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel

from agent.graph import invoke_agent, stream_agent
from agent.session_context import use_session
from agent.synthesis import synthesize_chat, synthesize_report
from rag.ingest import delete_session_documents

app = FastAPI(title="Multimodal Q&A Pro")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# session_id -> ALL images uploaded in that session (not just the most
# recent one — a user can upload several images in one session and ask
# questions that need more than one of them, same as multiple PDFs).
# Simple in-memory session store; matches the "ChromaDB is local/in-memory,
# resets on restart" note already called out in README's Known Limitations.
_SESSION_IMAGES: dict[str, list[str]] = {}


class ChatRequest(BaseModel):
    session_id: str
    query: str
    # Document Q&A tab (Section 3.2): "documents_only" makes search_documents
    # the preferred/first source, but still allows a search_web fallback when
    # search_documents comes back NOT_FOUND, or when the question needs
    # today's value and the uploaded document may be outdated.
    scope: Optional[str] = None
    # Image Studio tab: whether the cross-reference toggle is on, i.e.
    # whether the agent should also check uploaded documents for related
    # info while answering an image question. Previously accepted-but-ignored.
    cross_reference_documents: Optional[bool] = None


def _build_input_messages(req: ChatRequest) -> list[dict]:
    """If an image was uploaded earlier in this session, prepend that context
    so the routing prompt's rule 1 (image present -> describe_image first)
    has something concrete to act on.

    Also translates the frontend's `scope` and `cross_reference_documents`
    fields into explicit instructions in the message content. The agent's
    routing rules live in the system prompt in agent/graph.py (Vanshi's
    module), not here, so this is the interface-level way to influence
    per-request routing without changing that file: the same pattern already
    used for the image-path context below.
    """
    content = req.query
    image_paths = _SESSION_IMAGES.get(req.session_id) or []

    if image_paths:
        if len(image_paths) == 1:
            image_note = f"[An image was uploaded in this session at path: {image_paths[0]}."
        else:
            paths_list = "\n".join(f"  - {p}" for p in image_paths)
            image_note = (
                f"[{len(image_paths)} images were uploaded in this session, at these paths:\n"
                f"{paths_list}\n"
                f"Call describe_image once per path that's actually relevant to this "
                f"question — not necessarily all of them."
            )
        content = (
            f"{req.query}\n\n"
            f"{image_note} If relevant to this question, call describe_image with the "
            f"relevant path(s).]"
        )
        if req.cross_reference_documents:
            content += (
                "\n\n[Cross-reference mode is ON for this turn: after "
                "analyzing the image(s) with describe_image, also call "
                "search_documents to check whether the uploaded documents "
                "contain related or corroborating information, and call out "
                "any connection or discrepancy you find between the image(s) "
                "and the documents in your answer.]"
            )
        elif req.cross_reference_documents is False:
            content += (
                "\n\n[Cross-reference mode is OFF for this turn: answer using "
                "only describe_image. Do not call search_documents unless the "
                "question is clearly unrelated to the uploaded image(s).]"
            )

    if req.scope == "documents_only":
        # NOTE: "documents_only" now means "documents are the primary,
        # preferred source" rather than "web search is forbidden outright."
        # If the uploaded document is dated (e.g. from years ago) and the
        # question needs the current/up-to-date value, the agent's own
        # routing rules (agent/graph.py rule 3) already allow a search_web
        # fallback for explicitly current/live questions, or after
        # search_documents returns NOT_FOUND_IN_DOCUMENTS. This note just
        # reinforces the document-first ordering for this tab.
        content += (
            "\n\n[This question comes from the Document Q&A tab: prefer "
            "search_documents and use it FIRST. Only fall back to search_web "
            "if search_documents returns NOT_FOUND_IN_DOCUMENTS, or if the "
            "question is asking whether information in the (possibly dated) "
            "uploaded document is still current/accurate today — in that "
            "case call search_web too and note any discrepancy between the "
            "document and the current web result.]"
        )

    return [{"role": "user", "content": content}]


@app.post("/api/chat")
def chat(req: ChatRequest):
    """Chat Mode conversational response."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    with use_session(req.session_id, _SESSION_IMAGES.get(req.session_id)):
        result = invoke_agent(_build_input_messages(req))
    return {"session_id": req.session_id, "answer": synthesize_chat(result)}


@app.post("/api/chat/report")
def chat_report(req: ChatRequest):
    """Report Mode structured JSON response (schema per Section 3.2)."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    with use_session(req.session_id, _SESSION_IMAGES.get(req.session_id)):
        result = invoke_agent(_build_input_messages(req))
    report = synthesize_report(req.query, result)
    return {"session_id": req.session_id, "report": report}


@app.post("/api/upload/pdf")
async def upload_pdf(session_id: str = Form(...), file: UploadFile = File(...)):
    """Hands off to Dhanya's rag/ingest.py, returns confirmation + chunk count."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="only .pdf files are accepted")

    dest_path = os.path.join(UPLOAD_DIR, f"{session_id}_{file.filename}")
    contents = await file.read()
    with open(dest_path, "wb") as f:
        f.write(contents)

    try:
        from rag.ingest import ingest_pdf_result  # Dhanya's owned module
    except Exception as exc:
        # rag/ingest.py is currently an empty stub — fail the request clearly
        # rather than pretending ingestion succeeded.
        raise HTTPException(
            status_code=503,
            detail=f"PDF ingestion pipeline not yet available: {exc}",
        )

    try:
        # CRITICAL: ingest_pdf_result() is a long, blocking, synchronous call
        # (embedding model inference, ChromaDB writes, and -- worst case --
        # up to one sequential Groq network round-trip per OCR'd page).
        # Calling it directly here would block FastAPI's single asyncio
        # event loop for the entire duration: no other request (not even an
        # unrelated /api/chat call, a health check, or a second upload)
        # could be served until it finished, which is indistinguishable
        # from the server hanging even though it's technically "working."
        # asyncio.to_thread runs it in a worker thread instead, keeping the
        # event loop free the whole time.
        result = await asyncio.to_thread(
            ingest_pdf_result, dest_path, session_id=session_id
        )
    except ValueError as exc:
        # e.g. every page was blank, or OCR failed on every image-only page
        # (bad GROQ_API_KEY, vision model unavailable, etc).
        raise HTTPException(status_code=422, detail=str(exc))

    return {
        "session_id": session_id,
        "filename": file.filename,
        "chunks_ingested": result.chunk_count,
        "page_count": result.page_count,
        # Pages that had no real text layer and were transcribed via vision
        # OCR instead (e.g. PPT-exported slides that are flattened images).
        "ocr_page_count": result.ocr_page_count,
    }


@app.post("/api/upload/image")
async def upload_image(session_id: str = Form(...), file: UploadFile = File(...)):
    """Stores an image reference for the session, available to describe_image."""
    allowed_ext = (".jpg", ".jpeg", ".png", ".webp")
    if not file.filename.lower().endswith(allowed_ext):
        raise HTTPException(status_code=400, detail=f"only {allowed_ext} files are accepted")

    dest_path = os.path.join(UPLOAD_DIR, f"{session_id}_{uuid.uuid4().hex}_{file.filename}")
    contents = await file.read()
    with open(dest_path, "wb") as f:
        f.write(contents)

    # Append rather than overwrite — a session can upload multiple images
    # and have questions answered against any/all of them, same as PDFs.
    _SESSION_IMAGES.setdefault(session_id, []).append(dest_path)
    return {
        "session_id": session_id,
        "image_path": dest_path,
        "image_count": len(_SESSION_IMAGES[session_id]),
    }


@app.delete("/api/session/{session_id}")
def end_session(session_id: str):
    """
    Called when a session ends (page refresh/close, or an explicit 'reset'
    action) so that session's uploaded document chunks and image files are
    actually deleted rather than just becoming unreachable. Each page load
    generates a brand-new session_id client-side, so after this call that
    session_id has zero chunks and zero images — a genuinely fresh session,
    not just an inaccessible old one taking up space.
    """
    delete_session_documents(session_id)

    for image_path in _SESSION_IMAGES.pop(session_id, []):
        try:
            os.remove(image_path)
        except OSError:
            pass  # already gone / never existed — nothing to clean up

    return {"session_id": session_id, "status": "cleared"}


def _serialize_event(event: dict) -> str:
    """Turn one agent.stream() update into an SSE-friendly line. Only ever
    forwards what actually happened in the run — never fabricates ordering or
    content (Section 6.7)."""
    import json as _json

    lines = []
    for node_name, node_output in event.items():
        messages = node_output.get("messages", []) if isinstance(node_output, dict) else []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                lines.append({"type": "tool_result", "tool": msg.name, "content": str(msg.content)})
            elif isinstance(msg, AIMessage):
                if getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        lines.append({"type": "tool_call", "tool": tc["name"], "args": tc["args"]})
                if msg.content:
                    lines.append({"type": "ai_message", "node": node_name, "content": msg.content})
    return "\n".join(f"data: {_json.dumps(line)}" for line in lines) + "\n\n"


@app.get("/api/stream/{session_id}")
async def stream(
    session_id: str,
    query: str,
    scope: Optional[str] = None,
    cross_reference_documents: Optional[bool] = None,
):
    """SSE stream of the live reasoning trace, driven by the real
    agent.stream() output — not a simulated/fabricated trace."""

    def event_generator():
        messages = _build_input_messages(
            ChatRequest(
                session_id=session_id,
                query=query,
                scope=scope,
                cross_reference_documents=cross_reference_documents,
            )
        )
        with use_session(session_id, _SESSION_IMAGES.get(session_id)):
            for event in stream_agent(messages):
                serialized = _serialize_event(event)
                if serialized.strip():
                    yield serialized
        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# Static UI mount — same-origin, deliberately no CORS config (Section 7).
# Mounted LAST so it doesn't shadow the /api/* routes above.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
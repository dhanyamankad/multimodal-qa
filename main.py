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
import threading
import time
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
from agent.synthesis import strip_citation_artifacts, synthesize_chat, synthesize_report
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

# ---------------------------------------------------------------------------
# Conversation history — per session_id, a flat list of {"role", "content"}
# turns. This is what makes follow-up questions ("what team does HE play
# for?" after "Who is MS Dhoni?") actually work: previously every call to
# invoke_agent()/stream_agent() was seeded with ONLY the current message, so
# the agent had zero memory of anything said earlier in the same session —
# each turn ran as if it were a brand-new conversation. Now every /api/chat
# and /api/stream call prepends this session's prior turns before the new
# one, and appends the new turn once it's answered.
#
# Deliberately in-memory (matches _SESSION_IMAGES above) and deliberately
# simple: only the raw user query text and the final assistant answer text
# are stored — not tool calls/results — since that's enough context for
# follow-ups and keeps re-sent history small and stable across turns.
# ---------------------------------------------------------------------------
_SESSION_HISTORY: dict[str, list[dict]] = {}
# Cap how many prior turns get replayed, so a very long-running session
# doesn't grow the prompt (and recursion-limit risk) unboundedly. 8 turns
# (16 messages) is generous for follow-up-question continuity without
# ballooning every subsequent call.
_MAX_HISTORY_TURNS = 8


def _get_history(session_id: str) -> list[dict]:
    return _SESSION_HISTORY.get(session_id, [])


def _append_history(session_id: str, role: str, content: str) -> None:
    if not content:
        return
    history = _SESSION_HISTORY.setdefault(session_id, [])
    history.append({"role": role, "content": content})
    # Trim from the front, keeping the most recent turns.
    max_messages = _MAX_HISTORY_TURNS * 2
    if len(history) > max_messages:
        del history[: len(history) - max_messages]


# ---------------------------------------------------------------------------
# PDF ingest progress — per upload_id (generated client-side per upload, so
# multiple concurrent uploads in the same or different sessions don't clash),
# the latest known progress snapshot. Written from the worker thread running
# ingest_pdf_result() (see upload_pdf below), read by the SSE progress
# endpoint. Plain dict writes/reads are safe here without an extra lock —
# each upload_id is only ever written by its own single worker thread.
# ---------------------------------------------------------------------------
_INGEST_PROGRESS: dict[str, dict] = {}
_INGEST_PROGRESS_LOCK = threading.Lock()


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


def _build_full_messages(req: ChatRequest) -> list[dict]:
    """History-aware message list: this session's prior turns (plain
    user/assistant text, oldest first) followed by the newly-built current
    turn (which carries the image/scope/cross-reference notes). This is what
    gives the agent real conversational memory across turns instead of
    treating every request as a brand-new, context-free conversation."""
    return _get_history(req.session_id) + _build_input_messages(req)


@app.post("/api/chat")
def chat(req: ChatRequest):
    """Chat Mode conversational response."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    with use_session(req.session_id, _SESSION_IMAGES.get(req.session_id)):
        result = invoke_agent(_build_full_messages(req))
    answer = synthesize_chat(result)
    _append_history(req.session_id, "user", req.query)
    _append_history(req.session_id, "assistant", answer)
    return {"session_id": req.session_id, "answer": answer}


@app.post("/api/chat/report")
def chat_report(req: ChatRequest):
    """Report Mode structured JSON response (schema per Section 3.2)."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    with use_session(req.session_id, _SESSION_IMAGES.get(req.session_id)):
        result = invoke_agent(_build_full_messages(req))
    report = synthesize_report(req.query, result)
    # Reports aren't stored verbatim in history (they're structured JSON,
    # not a natural conversational turn), but a short marker is, so a later
    # follow-up question in Chat Mode ("what conflicts did that find?") still
    # has something to anchor to.
    _append_history(req.session_id, "user", req.query)
    _append_history(
        req.session_id,
        "assistant",
        f"[Generated an investigation report for: \"{req.query}\". "
        f"Conclusion: {report.get('conclusion', '')}]",
    )
    return {"session_id": req.session_id, "report": report}


@app.post("/api/upload/pdf")
async def upload_pdf(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    upload_id: Optional[str] = Form(None),
):
    """Hands off to Dhanya's rag/ingest.py, returns confirmation + chunk count.

    upload_id (optional, client-generated) namespaces the real-time progress
    exposed at GET /api/upload/progress/{upload_id}. If the client doesn't
    send one (e.g. an older frontend build), progress just won't be
    observable — ingestion itself is unaffected either way.
    """
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

    if upload_id:
        with _INGEST_PROGRESS_LOCK:
            _INGEST_PROGRESS[upload_id] = {
                "stage": "starting",
                "current": 0,
                "total": 0,
                "detail": "reading PDF",
                "done": False,
                "error": None,
            }

    def progress_cb(stage: str, current: int, total: int, detail: str) -> None:
        if not upload_id:
            return
        with _INGEST_PROGRESS_LOCK:
            _INGEST_PROGRESS[upload_id] = {
                "stage": stage,
                "current": current,
                "total": total,
                "detail": detail,
                "done": False,
                "error": None,
            }

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
            ingest_pdf_result,
            dest_path,
            session_id=session_id,
            progress_cb=progress_cb,
        )
    except ValueError as exc:
        # e.g. every page was blank, or OCR failed on every image-only page
        # (bad GROQ_API_KEY, vision model unavailable, etc).
        if upload_id:
            with _INGEST_PROGRESS_LOCK:
                _INGEST_PROGRESS[upload_id] = {
                    "stage": "error",
                    "current": 0,
                    "total": 0,
                    "detail": str(exc),
                    "done": True,
                    "error": str(exc),
                }
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 — surface to the progress stream too
        if upload_id:
            with _INGEST_PROGRESS_LOCK:
                _INGEST_PROGRESS[upload_id] = {
                    "stage": "error",
                    "current": 0,
                    "total": 0,
                    "detail": str(exc),
                    "done": True,
                    "error": str(exc),
                }
        raise

    if upload_id:
        with _INGEST_PROGRESS_LOCK:
            _INGEST_PROGRESS[upload_id] = {
                "stage": "done",
                "current": 1,
                "total": 1,
                "detail": "indexing complete",
                "done": True,
                "error": None,
            }

    return {
        "session_id": session_id,
        "filename": file.filename,
        "chunks_ingested": result.chunk_count,
        "page_count": result.page_count,
        # Pages that had no real text layer and were transcribed via vision
        # OCR instead (e.g. PPT-exported slides that are flattened images).
        "ocr_page_count": result.ocr_page_count,
    }


@app.get("/api/upload/progress/{upload_id}")
async def upload_progress(upload_id: str):
    """SSE stream of real-time indexing progress for one upload (Section
    'progressive progress bar' requirement). Driven entirely by the
    progress_cb snapshots written from the actual ingest worker thread in
    upload_pdf — never a simulated/timed fake progress bar. Ends the stream
    once that upload reports done=True (success or error), or after a
    generous timeout so a stream for an unknown/typo'd upload_id doesn't
    hang forever."""

    async def event_generator():
        import json as _json

        last_sent = None
        waited = 0.0
        poll_interval = 0.25
        max_wait_seconds = 600  # 10 min safety ceiling

        while waited < max_wait_seconds:
            with _INGEST_PROGRESS_LOCK:
                snapshot = _INGEST_PROGRESS.get(upload_id)
            if snapshot is not None and snapshot != last_sent:
                yield f"data: {_json.dumps(snapshot)}\n\n"
                last_sent = snapshot
                if snapshot.get("done"):
                    break
            await asyncio.sleep(poll_interval)
            waited += poll_interval

        # Clean up so the progress store doesn't grow unboundedly across
        # many uploads over a long-running server process.
        with _INGEST_PROGRESS_LOCK:
            _INGEST_PROGRESS.pop(upload_id, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


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


def _serialize_stream_item(mode: str, chunk, stream_state: dict) -> str:
    """Turn one item from stream_agent() (now stream_mode=["updates",
    "messages"], see agent/graph.py) into SSE-friendly line(s). Only ever
    forwards what actually happened in the run — never fabricates ordering,
    content, or timing (Section 6.7).
    """
    import json as _json

    lines = []
    if mode == "updates":
        for node_name, node_output in chunk.items():
            messages = node_output.get("messages", []) if isinstance(node_output, dict) else []
            for msg in messages:
                if isinstance(msg, ToolMessage):
                    lines.append({"type": "tool_result", "tool": msg.name, "content": str(msg.content)})
                elif isinstance(msg, AIMessage):
                    if getattr(msg, "tool_calls", None):
                        for tc in msg.tool_calls:
                            lines.append({"type": "tool_call", "tool": tc["name"], "args": tc["args"]})
                    if msg.content:
                        lines.append(
                            {
                                "type": "ai_message",
                                "node": node_name,
                                "content": strip_citation_artifacts(msg.content),
                            }
                        )
    elif mode == "messages":
        message_chunk, _metadata = chunk
        
        # CRITICAL FIX 2.0: LangGraph streams ALL chunks here, including raw 
        # Tool messages (the search snippets). We strictly ONLY want to stream 
        # tokens generated by the AI itself.
        if getattr(message_chunk, "type", "") != "ai":
            return ""

        msg_id = getattr(message_chunk, "id", None)
        content = getattr(message_chunk, "content", None)
        tool_calls = getattr(message_chunk, "tool_calls", [])
        tool_call_chunks = getattr(message_chunk, "tool_call_chunks", [])

        # Handle transitions between different messages
        if msg_id != stream_state["current_msg_id"]:
            # If the previous message was terminal, flush its buffered tokens
            if stream_state["current_msg_id"] is not None and stream_state["is_terminal_message"] and stream_state["token_buffer"]:
                for token in stream_state["token_buffer"]:
                    lines.append({"type": "token", "content": token})
            
            # Reset state for the new message
            stream_state["current_msg_id"] = msg_id
            stream_state["token_buffer"] = []
            stream_state["is_terminal_message"] = True

        # If we detect any tool calling activity, mark this message as intermediate
        if tool_calls or tool_call_chunks:
            stream_state["is_terminal_message"] = False

        # Buffer tokens instead of streaming immediately
        if content:
            stream_state["token_buffer"].append(content)

    return "\n".join(f"data: {_json.dumps(line)}" for line in lines) + ("\n\n" if lines else "")


@app.get("/api/stream/{session_id}")
async def stream(
    session_id: str,
    query: str,
    scope: Optional[str] = None,
    cross_reference_documents: Optional[bool] = None,
):
    """SSE stream of the live reasoning trace AND the real token-by-token
    final answer, driven entirely by the real agent.stream() output — never
    a simulated/fabricated trace or a fake typing animation.

    Also history-aware: this session's prior turns (see _SESSION_HISTORY)
    are prepended before the new one via _build_full_messages, and the new
    turn is appended once the run finishes — so a follow-up question asked
    right after ("what team does he play for?") still has the earlier turn
    ("Who is MS Dhoni?") in context instead of starting over from scratch.
    """
    req = ChatRequest(
        session_id=session_id,
        query=query,
        scope=scope,
        cross_reference_documents=cross_reference_documents,
    )

    async def event_generator():
        import json as _json
        import queue as _queue

        # NOTE ON SESSION-CONTEXT SAFETY:
        # This used to be a plain `def event_generator()` (sync generator)
        # with `with use_session(...):` wrapped around the `for item in
        # stream_agent(messages): ... yield ...` loop. That looks safe but
        # isn't: Starlette/anyio drive a sync generator behind an async
        # route by calling `next()` on it once per iteration via
        # `anyio.to_thread.run_sync`, and each of those calls can land on a
        # *different* worker thread from the threadpool. contextvars are
        # thread-scoped — `use_session`'s `.set()` only affects whichever
        # thread is running when it's called, and a suspended generator
        # does not carry "its" context across a resume on a different
        # thread. So mid-stream, a tool call (search_documents,
        # describe_image) could run on a thread where get_session_id()
        # still returns the default (None) or a stale value from whatever
        # else last ran on that thread — silently breaking session
        # isolation exactly in the one endpoint that streams.
        #
        # Fix: do ALL of the actual agent work — entering use_session(),
        # driving stream_agent() to completion, everything that touches
        # the session contextvar — inside a single dedicated worker
        # thread, started once via loop.run_in_executor. That thread's
        # context is set once and never has to survive being resumed on
        # a different thread. The worker pushes serialized SSE chunks
        # into a plain thread-safe queue.Queue; this async generator just
        # drains that queue and yields, which is safe because the
        # consumer side never touches session_id/contextvars at all.
        messages = _build_full_messages(req)
        q: "_queue.Queue" = _queue.Queue()
        _DONE = object()

        def worker():
            import json as _json
            last_ai_message: Optional[str] = None
            
            # Initialize streaming state for THIS specific request
            stream_state = {
                "token_buffer": [],
                "current_msg_id": None,
                "is_terminal_message": True
            }
            
            try:
                with use_session(session_id, _SESSION_IMAGES.get(session_id)):
                    for item in stream_agent(messages):
                        if isinstance(item, tuple) and len(item) == 2:
                            mode, chunk = item
                        elif isinstance(item, tuple) and len(item) == 3:
                            _namespace, mode, chunk = item
                        else:
                            mode, chunk = "updates", item
                        
                        if mode == "updates":
                            for _node_name, node_output in chunk.items():
                                node_messages = (
                                    node_output.get("messages", [])
                                    if isinstance(node_output, dict)
                                    else []
                                )
                                for msg in node_messages:
                                    if isinstance(msg, AIMessage) and msg.content:
                                        # FIX: Only track as final answer if there are NO tool calls
                                        if not getattr(msg, "tool_calls", None):
                                            last_ai_message = strip_citation_artifacts(msg.content)
                                            
                        # Pass stream_state to our updated serializer
                        serialized = _serialize_stream_item(mode, chunk, stream_state)
                        if serialized.strip():
                            q.put(("data", serialized))
            except Exception as exc:  # surface it instead of hanging the stream
                q.put(("error", str(exc)))
            finally:
                # FIX: Flush the very last message in the buffer if it was terminal
                if stream_state["is_terminal_message"] and stream_state["token_buffer"]:
                    final_lines = [{"type": "token", "content": t} for t in stream_state["token_buffer"]]
                    final_serialized = "\n".join(f"data: {_json.dumps(line)}" for line in final_lines) + "\n\n"
                    q.put(("data", final_serialized))
                
                q.put(("final", last_ai_message))
                q.put(_DONE)

        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, worker)

        last_ai_message: Optional[str] = None
        while True:
            kind_payload = await loop.run_in_executor(None, q.get)
            if kind_payload is _DONE:
                break
            kind, payload = kind_payload
            if kind == "data":
                yield payload
            elif kind == "error":
                yield f"data: {_json.dumps({'type': 'error', 'content': payload})}\n\n"
            elif kind == "final":
                last_ai_message = payload

        if last_ai_message:
            _append_history(session_id, "user", query)
            _append_history(session_id, "assistant", last_ai_message)

        # Final answer text is included here too (not just a bare "done")
        # so the frontend has one authoritative value to fall back on if,
        # for any reason, no token events came through.
        yield f"data: {_json.dumps({'type': 'done', 'content': last_ai_message})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# Static UI mount — same-origin, deliberately no CORS config (Section 7).
# Mounted LAST so it doesn't shadow the /api/* routes above.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
"""
FastAPI backend + API contract (PRD Section 3.1 / master PRD Section 7).

Serves the static UI from the same origin as the API (deliberate — no CORS
config needed, per Section 7) and exposes the chat/report/upload/stream
endpoints the frontend calls against.
"""

from __future__ import annotations

import os
import uuid
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel

from agent.graph import invoke_agent, stream_agent
from agent.synthesis import synthesize_chat, synthesize_report

app = FastAPI(title="Multimodal Q&A Pro")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# session_id -> most recently uploaded image path. Simple in-memory session
# store; matches the "ChromaDB is local/in-memory, resets on restart" note
# already called out in README's Known Limitations.
_SESSION_IMAGES: dict[str, str] = {}


class ChatRequest(BaseModel):
    session_id: str
    query: str
    # Document Q&A tab (Section 3.2): "documents_only" restricts the agent to
    # search_documents and forbids falling back to search_web, even on
    # NOT_FOUND_IN_DOCUMENTS. Previously accepted-but-ignored by Pydantic.
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
    image_path = _SESSION_IMAGES.get(req.session_id)

    if image_path:
        content = (
            f"{req.query}\n\n"
            f"[An image was uploaded in this session at path: {image_path}. "
            f"If relevant to this question, call describe_image with this path.]"
        )
        if req.cross_reference_documents:
            content += (
                "\n\n[Cross-reference mode is ON for this turn: after "
                "analyzing the image with describe_image, also call "
                "search_documents to check whether the uploaded documents "
                "contain related or corroborating information, and call out "
                "any connection or discrepancy you find between the image "
                "and the documents in your answer.]"
            )
        elif req.cross_reference_documents is False:
            content += (
                "\n\n[Cross-reference mode is OFF for this turn: answer using "
                "only describe_image. Do not call search_documents unless the "
                "question is clearly unrelated to the uploaded image.]"
            )

    if req.scope == "documents_only":
        content += (
            "\n\n[This question comes from the Document Q&A tab (scope="
            "documents_only): you must answer using ONLY search_documents. "
            "Do not call search_web under any circumstances. If "
            "search_documents returns NOT_FOUND_IN_DOCUMENTS, tell the user "
            "the answer was not found in the uploaded documents instead of "
            "falling back to a web search.]"
        )

    return [{"role": "user", "content": content}]


@app.post("/api/chat")
def chat(req: ChatRequest):
    """Chat Mode conversational response."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    result = invoke_agent(_build_input_messages(req))
    return {"session_id": req.session_id, "answer": synthesize_chat(result)}


@app.post("/api/chat/report")
def chat_report(req: ChatRequest):
    """Report Mode structured JSON response (schema per Section 3.2)."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
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
        from rag.ingest import ingest_pdf  # Dhanya's owned module
    except Exception as exc:
        # rag/ingest.py is currently an empty stub — fail the request clearly
        # rather than pretending ingestion succeeded.
        raise HTTPException(
            status_code=503,
            detail=f"PDF ingestion pipeline not yet available: {exc}",
        )

    chunk_count = ingest_pdf(dest_path, session_id=session_id)
    return {"session_id": session_id, "filename": file.filename, "chunks_ingested": chunk_count}


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

    _SESSION_IMAGES[session_id] = dest_path
    return {"session_id": session_id, "image_path": dest_path}


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
        for event in stream_agent(messages):
            serialized = _serialize_event(event)
            if serialized.strip():
                yield serialized
        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# Static UI mount — same-origin, deliberately no CORS config (Section 7).
# Mounted LAST so it doesn't shadow the /api/* routes above.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
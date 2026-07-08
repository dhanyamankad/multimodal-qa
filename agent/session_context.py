"""
agent/session_context.py

Carries the current request's session_id into tool execution without
making session_id an LLM-controlled tool argument. The agent's tools
(search_documents, describe_image) are invoked by the LLM itself — we never
want the model choosing or guessing a session_id, since that's exactly the
kind of value that must be enforced server-side to guarantee session
isolation (a prompt-injected or hallucinated session_id must not be able to
read another session's documents).

main.py sets this once per incoming request (before invoke_agent/stream_agent
runs) via `use_session(session_id)`; agent/tools.py reads it back with
`get_session_id()`.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Optional

_current_session_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_session_id", default=None
)

_current_allowed_images: contextvars.ContextVar[frozenset] = contextvars.ContextVar(
    "current_allowed_images", default=frozenset()
)


@contextmanager
def use_session(session_id: str, allowed_images: Optional[list] = None):
    id_token = _current_session_id.set(session_id)
    img_token = _current_allowed_images.set(frozenset(allowed_images or []))
    try:
        yield
    finally:
        _current_session_id.reset(id_token)
        _current_allowed_images.reset(img_token)


def get_session_id() -> Optional[str]:
    return _current_session_id.get()


def is_allowed_image(image_path: str) -> bool:
    return image_path in _current_allowed_images.get()

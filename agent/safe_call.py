"""
@safe_call — the single error-handling seam every tool call path runs through.

"""

from __future__ import annotations

import functools
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Callable, Optional

logger = logging.getLogger("agent.safe_call")


_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="safe_call")


class ToolFailure(Exception):
    """Internal marker so we can tell 'the tool raised' apart from 'the tool
    timed out' when logging — never re-raised past this module."""


def safe_call(fallback_message: Optional[str] = None, timeout: Optional[float] = None):
    """Decorator factory.

    Args:
        fallback_message: human-readable prefix returned to the agent (and thus
            visible in the trace) when the wrapped call fails. Keep this short
            and specific — the agent LLM reads this string and decides what to
            do next, so it needs to be usable as an observation, not just a log
            line.
        timeout: optional hard wall-clock timeout in seconds. If set, the call
            runs in a worker thread and is abandoned (not killed — Python can't
            forcibly kill a thread, but we stop waiting on it and answer the
            agent immediately) if it overruns.

    Usage:
        @tool
        @safe_call(fallback_message="Web search is temporarily unavailable.", timeout=8.0)
        def search_web(query: str) -> str:
            ...
    """

    def decorator(fn: Callable) -> Callable:
        tool_name = getattr(fn, "__name__", "tool")

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                if timeout is not None:
                    future = _EXECUTOR.submit(fn, *args, **kwargs)
                    try:
                        return future.result(timeout=timeout)
                    except FutureTimeoutError as exc:
                        raise ToolFailure(f"timed out after {timeout}s") from exc
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 — intentionally broad, this IS the safety net
                reason = str(exc) or exc.__class__.__name__
                logger.warning("safe_call: '%s' failed (%s)", tool_name, reason)
                prefix = fallback_message or f"{tool_name} is temporarily unavailable."
                
                return (
                    f"{prefix} (reason: {reason}). This source could not be reached "
                    f"right now — continue with whatever other information is "
                    f"available, and be upfront in the final answer that this "
                    f"source was unavailable rather than silently omitting it."
                )

        return wrapper

    return decorator

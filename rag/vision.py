"""
agent/vision.py

Single shared entry point for every Groq vision-model call in the app.

Previously this logic (client init, model constants, primary->fallback
retry) lived only inside agent/tools.py's describe_image tool. OCR for
image-only PDF pages (rag/ingest.py) needs the exact same vision call, so
it's factored out here rather than duplicated — keeping one place to update
if Groq deprecates a model again (see the dated note below).
"""

from __future__ import annotations

import os
import threading
import time

try:
    from groq import Groq
except ImportError:  # pragma: no cover - dependency should always be installed per requirements.txt
    Groq = None  # type: ignore

# Confirmed live against console.groq.com/docs/vision as of this build (2026-07-08):
VISION_MODEL_PRIMARY = "qwen/qwen3.6-27b"
# STATUS UPDATE (supersedes the PRD 3.0/14 deviation note): the PRD's
# original fallback string, meta-llama/llama-4-maverick-17b-128e-instruct,
# was deprecated by Groq on 2026-02-20 with no vision replacement, so it was
# a guaranteed-404, fail-closed-only branch (see PRD status log, 2026-07-07
# deviation entries). Fixed here: meta-llama/llama-4-scout-17b-16e-instruct
# is Groq's other currently-live vision model (console.groq.com/docs/vision
# confirms both qwen/qwen3.6-27b and this one are supported today), so the
# fallback is a real second model again, not a guaranteed failure. This
# deviates from the literal PRD instruction to keep the old string -- worth
# confirming with the team at the next sync -- but a fallback that can
# actually fall back seemed clearly better than one that can't ever succeed.
VISION_MODEL_FALLBACK = "meta-llama/llama-4-scout-17b-16e-instruct"

_client = Groq(api_key=os.environ.get("GROQ_API_KEY"), timeout=60.0) if Groq else None
# 60s per call: the Groq SDK has no timeout by default, which means a
# stalled connection (flaky network, Groq-side stall) would hang forever --
# and since rag/ingest.py's OCR loop makes one of these calls per page,
# sequentially, a single stuck call would look exactly like the whole
# upload silently freezing with no way to tell it apart from "still
# working." Bounding it means a bad call fails within 60s, gets caught by
# extract_pages' try/except, and that one page is logged and skipped
# instead of hanging the entire request indefinitely.


def get_vision_client():
    return _client


# ---------------------------------------------------------------------------
# Proactive pacing between Groq vision calls.
#
# Groq's per-minute rate limit is account-wide, not per-request, so firing
# 11 OCR calls back-to-back (one per PDF page) reliably trips a 429 partway
# through — the Groq SDK then retries with its own backoff (the 13s/25s/18s
# waits seen in testing). That backoff is reactive: it only kicks in AFTER a
# call has already failed, and grows with every subsequent failure, which is
# why an 11-page deck can end up taking minutes even though each individual
# OCR call is fast.
#
# This is a real, if partial, fix: space calls out up front so we ask for
# permission before Groq has to say no, instead of asking too fast and then
# paying a retry penalty. It can't lift the underlying rate limit itself
# (that's a Groq-side account limit — no client code can raise it), but it
# avoids the extra "already failed once" tax on every page, which in
# practice is most of what made this feel so slow.
# ---------------------------------------------------------------------------
_MIN_CALL_INTERVAL_SECONDS = 2.2
_pacing_lock = threading.Lock()
_last_call_at = 0.0


def _pace_call() -> None:
    global _last_call_at
    with _pacing_lock:
        now = time.monotonic()
        wait = _MIN_CALL_INTERVAL_SECONDS - (now - _last_call_at)
        if wait > 0:
            time.sleep(wait)
        _last_call_at = time.monotonic()


def vision_call(image_b64: str, question: str, image_format: str = "jpeg") -> str:
    """Ask the vision model a question about a base64-encoded image.

    Tries VISION_MODEL_PRIMARY first, then VISION_MODEL_FALLBACK on any
    failure. Raises the last exception if both fail — callers are
    responsible for catching it (agent tools go through @safe_call;
    rag/ingest.py catches it directly so one bad page doesn't kill the
    whole PDF's ingestion, see extract_pages()).
    """
    if _client is None:
        raise RuntimeError("Groq client not configured — check GROQ_API_KEY")

    last_exc: Exception | None = None
    for model_name in (VISION_MODEL_PRIMARY, VISION_MODEL_FALLBACK):
        try:
            _pace_call()
            response = _client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{image_format};base64,{image_b64}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1500,
            )
            return response.choices[0].message.content
        except Exception as exc:  # noqa: BLE001 — try next model, or re-raise below
            last_exc = exc
            continue
    raise last_exc  # both models failed

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

try:
    from groq import Groq
except ImportError:  # pragma: no cover - dependency should always be installed per requirements.txt
    Groq = None  # type: ignore

# Confirmed live against console.groq.com/docs/vision as of this build (2026-07-08):
VISION_MODEL_PRIMARY = "qwen/qwen3.6-27b"

VISION_MODEL_FALLBACK = "meta-llama/llama-4-scout-17b-16e-instruct"

_client = Groq(api_key=os.environ.get("GROQ_API_KEY"), timeout=60.0) if Groq else None

def get_vision_client():
    return _client


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

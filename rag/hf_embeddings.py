"""
rag/hf_embeddings.py

Drop-in replacement for langchain_huggingface.HuggingFaceEmbeddings that
calls Hugging Face's free serverless Inference API instead of loading
sentence-transformers/all-MiniLM-L6-v2 (and torch) locally.

Why this exists: torch + sentence-transformers add 700MB-1GB+ to the
container image and a comparable chunk of resident RAM the moment the
model loads -- which is what pushed this app over the 512MB free-tier
RAM ceiling on Render/Fly. Moving the embedding call out to HF's hosted
inference removes that weight entirely; only `requests` is needed.

Requires HF_TOKEN (a free Hugging Face access token -- Settings ->
Access Tokens on huggingface.co, "Read" scope is enough) set as an
environment variable / secret on whatever host runs this.

Interface matches what rag/ingest.py and rag/retrieve.py already call:
    embed_documents(list[str]) -> list[list[float]]
    embed_query(str) -> list[float]
"""

from __future__ import annotations

import logging
import os
import time
from typing import List

import requests

logger = logging.getLogger("rag.hf_embeddings")

HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
BATCH_SIZE = 32
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 2.0


class HFInferenceEmbeddings:
    def __init__(self, token: str | None = None):
        self._token = token or os.environ.get("HF_TOKEN")
        if not self._token:
            raise RuntimeError(
                "HF_TOKEN environment variable is not set. Get a free token at "
                "huggingface.co/settings/tokens and set it as HF_TOKEN."
            )
        self._headers = {"Authorization": f"Bearer {self._token}"}

    def _call(self, texts: List[str]) -> List[List[float]]:
        """
        Calls the feature-extraction endpoint with a batch of texts.
        Handles the classic HF Inference API cold-start behavior: a 503
        with an `estimated_time` while the model spins up on their side,
        which we retry with backoff rather than surface as a failure.
        """
        payload = {
            "inputs": texts,
            "options": {"wait_for_model": True},
        }
        backoff = INITIAL_BACKOFF_SECONDS
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    HF_API_URL, headers=self._headers, json=payload, timeout=60
                )
                if resp.status_code == 200:
                    data = resp.json()
                    # feature-extraction on a sentence-transformers model
                    # returns one vector per input string directly.
                    return data
                if resp.status_code == 503:
                    logger.info(
                        "HF Inference API cold start (attempt %d/%d), retrying in %.1fs",
                        attempt, MAX_RETRIES, backoff,
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                resp.raise_for_status()
            except requests.RequestException as exc:
                last_error = exc
                logger.warning("HF Inference API call failed (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)
                time.sleep(backoff)
                backoff *= 2
        raise RuntimeError(f"HF Inference API failed after {MAX_RETRIES} attempts: {last_error}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        results: List[List[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            results.extend(self._call(batch))
        return results

    def embed_query(self, text: str) -> List[float]:
        return self._call([text])[0]

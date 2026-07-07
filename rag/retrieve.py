"""
rag/retrieve.py

Confidence-threshold retrieval logic (PRD Section 6.1).

This is the module that prevents hallucination when no relevant document
exists for a query: below-threshold results are treated as "not found"
and MUST NOT be forced into an answer. This is what makes:
  - PS5 test scenario 1 (pure doc question -> only search_documents fires)
  - PS4-style scenario 6/7 (doc-only vs. no-answer-in-docs)
pass cleanly, since the agent's routing logic (owned by Vanshi, in
agent/tools.py / agent/graph.py) depends on getting an honest signal here
rather than a chunk that's technically returned but semantically irrelevant.

INTERFACE CONTRACT (this is what Vanshi's `search_documents` tool consumes --
do not change this shape without syncing with her first):

    retrieve(query: str, top_k: int = 4) -> RetrievalResponse

    RetrievalResponse.found: bool
        False means "nothing above threshold" -- the tool should tell the
        agent "not found in documents," never fabricate an answer.

    RetrievalResponse.chunks: List[RetrievedChunk]
        Empty when found=False. Each chunk carries `filename` and
        `page_number` so citations (Chat Mode and Report Mode) always have
        a real, specific source per PRD Section 3.2's rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from rag.ingest import get_ingestor


# Chroma's default distance metric is cosine distance (0 = identical,
# 2 = opposite). We convert to a similarity score in [0, 1] and require
# a minimum similarity for a chunk to count as "relevant" rather than
# just "closest available, even if unrelated."
#
# THRESHOLD CALIBRATION (empirically tested, not a guess):
# all-MiniLM-L6-v2 cosine similarities run much lower in absolute terms
# than intuition suggests, especially on short resume/bio-style chunks.
# Real test against a sample resume PDF:
#   - Genuinely relevant chunk (query: "what is the email id" -> chunk
#     containing the actual email address): similarity ~0.138-0.171
#   - Genuinely irrelevant query ("what is the capital of France?"):
#     similarity ~ -0.075 to -0.023
# The original 0.35 threshold was set before this test and would have
# rejected every real answer. 0.08 sits in the gap between the two
# clusters above, with margin on both sides. Re-validate this if the
# document style changes significantly (e.g. long-form reports vs. short
# resumes may shift the relevant-chunk cluster).
SIMILARITY_THRESHOLD = 0.08
DEFAULT_TOP_K = 4


@dataclass
class RetrievedChunk:
    text: str
    filename: str
    page_number: int
    similarity: float


@dataclass
class RetrievalResponse:
    found: bool
    chunks: List[RetrievedChunk] = field(default_factory=list)
    query: str = ""

    def as_context_string(self) -> str:
        """
        Flatten chunks into a single string suitable for stuffing into an
        LLM prompt, each chunk tagged with its citation so the synthesis
        layer (Vanshi's agent/synthesis.py) can attribute claims correctly
        in both Chat Mode and Report Mode.
        """
        if not self.found:
            return "No relevant information found in uploaded documents."
        parts = []
        for c in self.chunks:
            parts.append(f"[{c.filename}, p.{c.page_number}]\n{c.text}")
        return "\n\n".join(parts)


def _distance_to_similarity(distance: float) -> float:
    """
    Convert Chroma cosine distance (range ~0-2) to a similarity score in
    [0, 1], where 1 = identical. Clamped defensively in case of
    floating-point edge cases from the underlying index.
    """
    similarity = 1.0 - (distance / 2.0)
    return max(0.0, min(1.0, similarity))


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
) -> RetrievalResponse:
    """
    Query ChromaDB for the top_k most similar chunks to `query`, then
    filter to only those above `threshold`.

    Returns found=False (never a forced/best-effort answer) when either:
      - the collection is empty (nothing has been ingested yet), or
      - every candidate falls below the similarity threshold.
    """
    ingestor = get_ingestor()

    if ingestor.collection.count() == 0:
        return RetrievalResponse(found=False, query=query)

    query_embedding = ingestor.embeddings.embed_query(query)

    results = ingestor.collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, max(ingestor.collection.count(), 1)),
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    chunks: List[RetrievedChunk] = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        similarity = _distance_to_similarity(dist)
        if similarity >= threshold:
            chunks.append(
                RetrievedChunk(
                    text=doc,
                    filename=meta.get("filename", "unknown"),
                    page_number=meta.get("page_number", -1),
                    similarity=similarity,
                )
            )

    if not chunks:
        return RetrievalResponse(found=False, query=query)

    return RetrievalResponse(found=True, chunks=chunks, query=query)

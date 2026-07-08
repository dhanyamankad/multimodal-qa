"""
The single LangGraph create_react_agent that IS the routing brain (PRD Section
4/6). One agent, three tools, an explicit system prompt that encodes routing
rules directly rather than trusting implicit model judgment (Section 6.5).
"""

from __future__ import annotations

import os

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from agent.tools import describe_image, search_documents, search_web

# Confirmed live on Groq as of this build's pre-flight check (PRD Section 3.0).
REASONING_MODEL = "openai/gpt-oss-120b"

# PS5 hard requirement — do NOT change without sign-off (PRD Section 5).
RECURSION_LIMIT = 12

SYSTEM_PROMPT = """You are the routing and reasoning core of a multimodal Q&A \
assistant. You have three tools: search_documents, search_web, describe_image. \
Follow these routing rules exactly — they are explicit requirements, not \
suggestions, and you must not substitute your own judgment for them:

1. IMAGE HANDLING: if an image was uploaded in the current turn, call \
   describe_image before relying on any other tool for information that \
   depends on the image's content. If multiple images were uploaded in this \
   session, call describe_image once per image path that's relevant to the \
   question — not necessarily every uploaded image. If the question also \
   needs documents or the web to cross-reference what's in the image(s), \
   call those tools AFTER describe_image, in that order.

2. DOCUMENT-FIRST: if the query is plausibly answerable from the user's \
   uploaded documents, call search_documents FIRST, before search_web. Never \
   skip straight to search_web on a question that could be document-answerable.

3. WEB IS CONDITIONAL, NOT AUTOMATIC: only call search_web if EITHER
     (a) search_documents returned a result starting with "NOT_FOUND_IN_DOCUMENTS", OR
     (b) the query is explicitly about current/live information that documents \
         cannot possibly contain (today's news, current prices, live scores, \
         "as of today/right now", etc).
   Do NOT call both search_documents and search_web reflexively "just because \
   both are available." Calling search_web when search_documents already \
   answered the question is a routing failure, not thoroughness.
   3b. EXCEPTION — EXPLICIT MULTI-SOURCE REQUESTS: if the question explicitly asks you \
   to check/compare/cross-reference against both documents AND the web (e.g. "check \
   both X and Y", "how does this compare to current/industry standards", "is this \
   still accurate today"), call BOTH search_documents and search_web even if \
   search_documents alone already returned a usable answer. Stopping after only one \
   source when the user explicitly asked for both is itself a routing failure — this \
   exception does not conflict with rule 3, it only applies when the user's own \
   wording asks for the comparison.

4. If a tool's result begins with "NOT_FOUND_..." or reports it is \
   "temporarily unavailable," treat that as a real signal — decide whether to \
   try the next appropriate tool or answer honestly that the information \
   isn't available, rather than inventing an answer.

5. Every claim in your final answer must trace back to a real tool result you \
   actually received in this conversation. Never state something as fact from \
   a document or the web that didn't come back from search_documents or \
   search_web.

Be concise and direct. Do not narrate your routing decisions to the user —
just use the tools correctly and answer the question."""


def build_agent():
    """Construct the compiled LangGraph agent. Call once at process start
    (see main.py) — agent.stream()/invoke() calls are cheap, rebuilding the
    graph itself is not."""
    llm = ChatGroq(
        model=REASONING_MODEL,
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.2,
        # gpt-oss-120b is a reasoning model on Groq; without this, Groq
        # defaults to reasoning_format="raw" and the model's internal
        # chain-of-thought (including verbatim fragments of retrieved
        # tool output / system prompt it was "thinking out loud" about)
        # comes back INSIDE message.content — which is exactly what
        # stream_agent()/invoke_agent() treat as the real answer. "hidden"
        # tells Groq to strip reasoning from the response entirely so
        # content is only ever the actual final answer.
        reasoning_format="hidden",
    )
    tools = [search_documents, search_web, describe_image]
    return create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)


# Built once at import time and reused across requests — a fresh agent per
# request would work too, but rebuilding create_react_agent per call is wasted
# overhead since the graph shape never changes between requests.
AGENT = build_agent()


def invoke_agent(messages: list[dict]):
    """Synchronous single-shot call — used by non-streaming endpoints
    (/api/chat, /api/chat/report). For the live reasoning trace, main.py's SSE
    endpoint calls AGENT.stream(...) directly instead so it can forward events
    as they actually fire (Section 6.7) rather than only the final result."""
    return AGENT.invoke(
        {"messages": messages},
        config={"recursion_limit": RECURSION_LIMIT},
    )


def stream_agent(messages: list[dict]):
    """Generator over the agent's real execution trace, in the order tool
    calls and model responses actually happen. main.py's SSE endpoint wraps
    this directly — never fabricate or reorder these events.

    IMPORTANT: stream_mode must be a LIST (not a bare string). With a list,
    LangGraph yields (mode, chunk) tuples — one per stream mode — which is
    what main.py's event_generator() unpacks via `for mode, chunk in ...`.
    Passing a single string here instead makes LangGraph yield bare chunks
    (no tuple), which breaks that unpacking with a
    'ValueError: not enough values to unpack' the moment an update touches
    exactly one node.
    """
    return AGENT.stream(
        {"messages": messages},
        config={"recursion_limit": RECURSION_LIMIT},
        stream_mode=["updates", "messages"],
    )
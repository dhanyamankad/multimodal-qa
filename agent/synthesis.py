"""
Synthesis layer — turns the agent's raw run (messages + tool results) into
either a Chat Mode answer or a Report Mode structured JSON document.

Chat Mode: the agent's own final message IS the answer. No extra LLM call.
Report Mode: a dedicated second-pass LLM call, fed the ACTUAL tool outputs
collected from the run (not re-derived from the chat answer), so every
`findings` entry traces to something real (Section 3.5) and `conflicts` comes
from an explicit comparison step rather than implicit judgment (Section 16).
"""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_groq import ChatGroq

REASONING_MODEL = "llama-3.3-70b-versatile"

_report_llm = ChatGroq(
    model=REASONING_MODEL,
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0,
)

REPORT_JSON_SCHEMA = {
    "findings": [
        {
            "claim": "...",
            "source_type": "document | web",
            "source_detail": "filename.pdf, p.4  OR  domain.com",
        }
    ],
    "conflicts": [
        {
            "topic": "...",
            "document_claim": "...",
            "web_claim": "...",
            "note": "...",
        }
    ],
    "conclusion": "...",
}


def synthesize_chat(agent_result: dict[str, Any]) -> str:
    """Chat Mode: natural conversational synthesis, sources implied not
    itemized. The agent's own last AIMessage already is the synthesis — this
    function's job is just to extract it safely."""
    messages = agent_result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return "I wasn't able to produce an answer for that — please try rephrasing the question."


def _extract_tool_outputs(agent_result: dict[str, Any]) -> dict[str, list[str]]:
    """Pull real ToolMessage contents out of the finished run, grouped by
    which tool produced them. This is the ground truth Report Mode is built
    from — never re-derive findings from the chat-mode prose instead."""
    grouped: dict[str, list[str]] = {"search_documents": [], "search_web": [], "describe_image": []}
    for msg in agent_result.get("messages", []):
        if isinstance(msg, ToolMessage) and msg.name in grouped:
            grouped[msg.name].append(str(msg.content))
    return grouped


def _build_report_prompt(query: str, tool_outputs: dict[str, list[str]]) -> str:
    doc_block = "\n---\n".join(tool_outputs["search_documents"]) or "(search_documents was not called)"
    web_block = "\n---\n".join(tool_outputs["search_web"]) or "(search_web was not called)"
    image_block = "\n---\n".join(tool_outputs["describe_image"]) or "(describe_image was not called)"

    return f"""You are building a structured investigation report for the question:
"{query}"

You have three raw source blocks below. Build the report in exactly two steps:

STEP 1 — FINDINGS: extract discrete factual claims from each block, tagging
each with its real source_type ("document" or "web") and a specific
source_detail (exact filename+page for documents, exact domain for web).
Only include a finding if it is directly traceable to one of these blocks —
never invent or infer an attribution.

STEP 2 — EXPLICIT CONFLICT CHECK: compare the document findings against the
web findings sub-topic by sub-topic. Populate `conflicts` where a document
claim and a web claim make genuinely incompatible statements about the SAME
sub-topic (e.g. different version numbers, different dates, contradictory
yes/no facts). This explicitly INCLUDES the case where a document states an
older fact and the web reports a newer/current fact that has since changed
(e.g. a document says X was true as of 2022, the web says Y is true as of
now) — a document being outdated relative to current reality IS a genuine
conflict for this purpose, not a "different time periods, no conflict"
situation, especially when the original question asks about the CURRENT
state of something. Do not report a conflict for topics only one source
covers, and do not report a conflict for claims that are merely differently
worded but not actually contradictory. A missed real conflict and a
false-positive conflict are equally bad — be conservative, but an
outdated-vs-current disagreement is not a borderline case, it is a conflict.


--- DOCUMENT SOURCE BLOCK ---
{doc_block}

--- WEB SOURCE BLOCK ---
{web_block}

--- IMAGE ANALYSIS BLOCK ---
{image_block}

Respond with ONLY a single JSON object, no markdown fences, no preamble,
matching exactly this shape:
{json.dumps(REPORT_JSON_SCHEMA, indent=2)}

If there are no genuine conflicts, return an empty list for "conflicts". If a
block was not called, do not fabricate findings for it.
"""


def _parse_json_safely(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fail closed with a clearly-flagged, still-valid-shape report rather
        # than crashing the endpoint or silently returning malformed JSON.
        return {
            "findings": [],
            "conflicts": [],
            "conclusion": (
                "Report generation failed to produce valid structured output. "
                "Raw model output could not be parsed as JSON."
            ),
        }


def synthesize_report(query: str, agent_result: dict[str, Any]) -> dict[str, Any]:
    """Report Mode: structured JSON exactly per PRD Section 3.2 schema."""
    tool_outputs = _extract_tool_outputs(agent_result)
    prompt = _build_report_prompt(query, tool_outputs)
    response = _report_llm.invoke(prompt)
    return _parse_json_safely(response.content)

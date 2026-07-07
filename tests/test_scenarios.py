"""
Scripted PS5 + self-imposed test scenarios (PRD Section 3.6 / 11).

These exercise the real agent graph end-to-end (create_react_agent -> Groq),
with the underlying tool implementations monkeypatched to deterministic fakes
so scenarios are reproducible without depending on live document content or
the actual DuckDuckGo/Groq-vision network calls succeeding or failing on a
given day. The ROUTING DECISIONS (which tools fire, in what order) are made
for real by the live reasoning model — that's the thing under test.

Requires GROQ_API_KEY (real network call to Groq for the reasoning model
itself) — scenarios are skipped automatically if it's not set, e.g. in a
sandboxed environment with no outbound access to api.groq.com.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, ToolMessage

pytestmark = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set — these scenarios need a live call to the reasoning model",
)


def _tool_names_called(agent_result) -> list[str]:
    return [m.name for m in agent_result["messages"] if isinstance(m, ToolMessage)]


def _final_text(agent_result) -> str:
    for m in reversed(agent_result["messages"]):
        if isinstance(m, AIMessage) and m.content:
            return m.content
    return ""


# --- Scenario 1: Pure doc question -> only search_documents fires ----------
def test_01_pure_doc_question_only_search_documents_fires(monkeypatch):
    import agent.tools as tools

    monkeypatch.setattr(
        tools,
        "retrieve_chunks",
        lambda query, threshold=0.35: [
            {"text": "The refund window is 30 days.", "filename": "policy.pdf", "page": 2, "score": 0.9}
        ],
    )

    from agent.graph import invoke_agent

    result = invoke_agent([{"role": "user", "content": "What is the refund window per the policy document?"}])
    called = _tool_names_called(result)
    assert "search_documents" in called
    assert "search_web" not in called


# --- Scenario 2: Image + doc cross-reference, trace order matters ----------
def test_02_image_then_doc_cross_reference_order(monkeypatch):
    import agent.tools as tools

    monkeypatch.setattr(tools, "_vision_call", lambda model, b64, q: "A blue ceramic mug with a chip on the handle.")
    monkeypatch.setattr(
        tools,
        "retrieve_chunks",
        lambda query, threshold=0.35: [
            {"text": "Product catalog: blue ceramic mug, SKU 1183.", "filename": "catalog.pdf", "page": 5, "score": 0.8}
        ],
    )
    # describe_image needs a real file on disk to open() — provide a tiny stub image.
    import tempfile
    tmp_path = os.path.join(tempfile.gettempdir(), "test_scenario_2.jpg")
    with open(tmp_path, "wb") as f:
        f.write(b"\xff\xd8\xff\xe0fake-jpeg-bytes")

    from agent.graph import invoke_agent

    result = invoke_agent(
        [
            {
                "role": "user",
                "content": (
                    f"[An image was uploaded in this session at path: {tmp_path}. "
                    f"If relevant to this question, call describe_image with this path.] "
                    f"What product in our catalog does this image match?"
                ),
            }
        ]
    )
    called = _tool_names_called(result)
    assert "describe_image" in called
    assert "search_documents" in called
    assert called.index("describe_image") < called.index("search_documents")


# --- Scenario 3: Current-info question, no relevant docs ------------------
def test_03_current_info_no_docs_falls_back_to_web(monkeypatch):
    import agent.tools as tools

    monkeypatch.setattr(tools, "retrieve_chunks", lambda query, threshold=0.35: [])

    class FakeDDG:
        def run(self, query):
            return "Today's headline: markets rallied on rate-cut expectations."

    monkeypatch.setattr("langchain_community.tools.DuckDuckGoSearchRun", lambda: FakeDDG())

    from agent.graph import invoke_agent

    result = invoke_agent([{"role": "user", "content": "What is today's top financial news headline?"}])
    called = _tool_names_called(result)
    assert "search_web" in called


# --- Scenario 4: Simulated web search timeout -> reported gracefully ------
def test_04_web_search_timeout_reported_gracefully(monkeypatch):
    import time

    import agent.tools as tools

    monkeypatch.setattr(tools, "retrieve_chunks", lambda query, threshold=0.35: [])

    class SlowDDG:
        def run(self, query):
            time.sleep(20)  # exceeds the 8s safe_call timeout on search_web
            return "should never get here"

    monkeypatch.setattr("langchain_community.tools.DuckDuckGoSearchRun", lambda: SlowDDG())

    # Call the wrapped tool function directly to isolate safe_call's behavior
    # from full-agent latency in this specific test.
    fallback = tools.search_web.invoke({"query": "current weather in Paris"})
    assert "temporarily unavailable" in fallback.lower()
    assert "timed out" in fallback.lower() or "reason" in fallback.lower()


# --- Scenario 5: Local cold-start end-to-end equivalent --------------------
def test_05_cold_start_end_to_end(monkeypatch):
    import agent.tools as tools

    monkeypatch.setattr(
        tools,
        "retrieve_chunks",
        lambda query, threshold=0.35: [
            {"text": "Founded in 2019, headquartered in Rajkot.", "filename": "about.pdf", "page": 1, "score": 0.7}
        ],
    )
    from agent.graph import invoke_agent

    result = invoke_agent([{"role": "user", "content": "Where is the company headquartered?"}])
    assert _final_text(result)  # produced *some* real answer end-to-end


# --- Scenario 6: Doc-only question -> no unnecessary web search -----------
def test_06_doc_only_no_unnecessary_web_search(monkeypatch):
    import agent.tools as tools

    monkeypatch.setattr(
        tools,
        "retrieve_chunks",
        lambda query, threshold=0.35: [
            {"text": "Warranty period is 12 months from purchase.", "filename": "warranty.pdf", "page": 1, "score": 0.85}
        ],
    )
    from agent.graph import invoke_agent

    result = invoke_agent([{"role": "user", "content": "How long is the warranty according to the document?"}])
    called = _tool_names_called(result)
    assert "search_web" not in called


# --- Scenario 7: No answer in docs, answerable on web ----------------------
def test_07_no_doc_answer_clean_fallback_to_web(monkeypatch):
    import agent.tools as tools

    monkeypatch.setattr(tools, "retrieve_chunks", lambda query, threshold=0.35: [])

    class FakeDDG:
        def run(self, query):
            return "The current exchange rate is 1 USD = 83.2 INR."

    monkeypatch.setattr("langchain_community.tools.DuckDuckGoSearchRun", lambda: FakeDDG())

    from agent.graph import invoke_agent

    result = invoke_agent([{"role": "user", "content": "What is today's USD to INR exchange rate?"}])
    called = _tool_names_called(result)
    assert "search_web" in called
    assert _final_text(result)


# --- Scenario 8: Outdated doc fact vs current web fact -> conflicts -------
def test_08_conflict_between_doc_and_web(monkeypatch):
    import agent.tools as tools

    monkeypatch.setattr(
        tools,
        "retrieve_chunks",
        lambda query, threshold=0.35: [
            {"text": "The current CEO is Jane Doe (as of 2022).", "filename": "handbook.pdf", "page": 3, "score": 0.8}
        ],
    )

    class FakeDDG:
        def run(self, query):
            return "As of 2026, the company's CEO is John Smith, per the company's newsroom."

    monkeypatch.setattr("langchain_community.tools.DuckDuckGoSearchRun", lambda: FakeDDG())

    from agent.graph import invoke_agent
    from agent.synthesis import synthesize_report

    query = "Who is the current CEO — check both the handbook and the web?"
    result = invoke_agent([{"role": "user", "content": query}])
    report = synthesize_report(query, result)
    print("\n--- DEBUG: tools called ---", _tool_names_called(result))
    print("--- DEBUG: full report ---", report)
    assert len(report.get("conflicts", [])) >= 1

# --- Scenario 9: Multi-part question needing both sources -----------------
def test_09_multi_part_question_correct_attribution(monkeypatch):
    import agent.tools as tools

    monkeypatch.setattr(
        tools,
        "retrieve_chunks",
        lambda query, threshold=0.35: [
            {"text": "Our return policy allows returns within 30 days.", "filename": "policy.pdf", "page": 4, "score": 0.9}
        ],
    )

    class FakeDDG:
        def run(self, query):
            return "Industry-standard return windows in 2026 average 45 days, per NRF."

    monkeypatch.setattr("langchain_community.tools.DuckDuckGoSearchRun", lambda: FakeDDG())

    from agent.graph import invoke_agent
    from agent.synthesis import synthesize_report

    query = "What is our return policy, and how does it compare to current industry standards?"
    result = invoke_agent([{"role": "user", "content": query}])
    report = synthesize_report(query, result)
    findings = report.get("findings", [])
    assert any(f.get("source_type") == "document" for f in findings)
    assert any(f.get("source_type") == "web" for f in findings)

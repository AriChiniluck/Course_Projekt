from functools import lru_cache
import json
import threading

from langchain.agents import create_agent
from langchain_core.tools import tool
from langfuse import observe, propagate_attributes

from observability import get_langfuse_handler
from tools import web_search, read_url, knowledge_search
from config import build_chat_model, settings, get_research_system_prompt

_research_lock = threading.Lock()


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content) if content else ""


def _build_research_workflow(sources: list[str]) -> str:
    normalized = {str(item).strip().lower() for item in (sources or []) if str(item).strip()}
    if {"knowledge_base", "web"}.issubset(normalized):
        return (
            "Required workflow: first use exactly one knowledge_search to establish the baseline from the internal knowledge base only. "
            "Summarize what the internal materials say, identify any obvious gaps or potentially stale points, and only then use one web_search to verify freshness or detect what has changed. "
            "Use read_url only for the single most relevant external source if a snippet is insufficient. Do not start with web search and do not mix web claims into the baseline before checking the internal materials."
        )
    if "knowledge_base" in normalized and "web" not in normalized:
        return (
            "Use only the internal knowledge base unless the user explicitly asks you to go beyond it because the internal evidence is insufficient."
        )
    if "web" in normalized and "knowledge_base" not in normalized:
        return "Use the web as the primary source of evidence."
    return "Prefer the internal knowledge base first, then use the web only if it adds necessary verification or freshness."


@lru_cache(maxsize=1)
def _build_research_agent():
    return create_agent(
        model=build_chat_model(temperature=0.2, model=settings.researcher_model),
        tools=[web_search, read_url, knowledge_search],
        system_prompt=get_research_system_prompt(),
    )


def get_research_agent():
    """Return (lazily creating) the shared research agent. Thread-safe on first initialisation."""
    with _research_lock:
        return _build_research_agent()


@tool
@observe(name="research_tool")
def research(plan: str) -> str:
    """Execute the research plan and return concise findings with sources."""
    plan_text = str(plan or "").strip()
    if not plan_text:
        return "Research plan is empty."

    # First calls arrive as a ResearchPlan JSON from the Planner.
    # Revision calls arrive as plain text from the middleware — pass through as-is.
    try:
        sources: list[str] = []
        parsed = json.loads(plan_text)
        if isinstance(parsed, dict) and "goal" in parsed:
            goal = parsed.get("goal", "")
            queries = parsed.get("search_queries") or []
            sources = parsed.get("sources_to_check") or []
            output_format = parsed.get("output_format", "")
            parts = [f"Research goal: {goal}"]
            if queries:
                parts.append("Search queries to execute:\n" + "\n".join(f"- {q}" for q in queries))
            if sources:
                parts.append("Sources to consult: " + ", ".join(sources))
            if output_format:
                parts.append(f"Expected output format: {output_format}")
            plan_text = "\n\n".join(parts)
    except (json.JSONDecodeError, TypeError):
        pass  # plain text revision plan — use as-is

    research_request = (
        "Execute the following research plan and gather evidence.\n\n"
        f"{plan_text}\n\n"
        "Important: answer in the same language as the user's request. Use the local knowledge base first for course topics. Use as few tool calls as possible: normally 1 knowledge_search, 1 web_search, and at most 1 read_url only if absolutely necessary. Do not repeat similar searches. If the request refers to a future year or future state beyond today, do NOT invent facts: explicitly frame the answer as a forecast, outlook, or evidence-based projection using the latest verified information available today. After gathering enough evidence, stop and write the final synthesis immediately. Keep source metadata in the form 'Source / page / Relevance' when available. Unless the user explicitly asks for a long report, prefer about 220-320 words or 4-6 concise bullets.\n\n"
        f"{_build_research_workflow(sources)}"
    )
    try:
        with propagate_attributes(metadata={"agent_name": "researcher", "stage": "evidence_collection"}):
            result = get_research_agent().invoke(
                {"messages": [{"role": "user", "content": research_request}]},
                config={
                    "recursion_limit": settings.graph_recursion_limit,
                    "callbacks": [get_langfuse_handler()],
                },
            )
    except Exception as exc:
        return f"Research agent failed: {exc}"

    messages = result.get("messages", [])
    if not messages:
        return "Researcher did not return any findings. Stopping research."

    content = _content_to_text(getattr(messages[-1], "content", ""))
    # Stop condition: if agent returns empty, generic, or no new evidence
    stop_phrases = [
        "no new evidence", "no new information", "nothing found", "cannot find", "did not find",
        "no relevant sources", "no further revision", "insufficient data", "not enough data",
        "no findings", "no results", "no sources found"
    ]
    content_stripped = content.strip()
    # Only treat as a terminal stop when the content is short (< 300 chars).
    # A longer response that mentions "no results for X but found Y via Z" contains
    # useful evidence and must NOT be suppressed by a partial phrase match.
    is_stop_phrase_hit = any(phrase in content.lower() for phrase in stop_phrases)
    if not content_stripped or (is_stop_phrase_hit and len(content_stripped) < 300):
        return f"Researcher could not find new evidence or complete the revision. Stopping research.\n\n{content}"

    return content

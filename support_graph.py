"""LangGraph StateGraph — Telekom Customer Support (Routing pattern).

Architecture (Anthropic Routing pattern):

    START
      │
      ▼
  [router]  ← ClassificationOutput (structured output, no tools)
      │
      ├─ category="product"  ──▶ [docs]       search_customer + search_telekom_kb
      ├─ category="general"  ──▶ [websearch]  lookup_tariff_telekom
      └─ category="critical" ──▶ [escalation] send_escalation_email
                                       ▲
      docs / websearch ────────────────┘
      (fallback when agent returns NEEDS_ESCALATION signal)
                                       │
                                      END

Langfuse CallbackHandler is injected into every agent invocation so all
LLM calls are traced (latency, tokens, metadata per agent).
"""
from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

try:
    from langgraph.checkpoint.sqlite import SqliteSaver as _SqliteSaver
    _SQLITE_AVAILABLE = True
except ImportError:
    _SQLITE_AVAILABLE = False

# Agent modules are imported lazily inside node functions to avoid loading
# transformers / sentence-transformers at module import time (slow on Windows).
from observability import get_langfuse_handler

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class SupportState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    classification: dict          # ClassificationOutput.model_dump()
    session_id: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ESCALATION_SIGNAL = "NEEDS_ESCALATION:"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _needs_escalation(messages: list[BaseMessage]) -> bool:
    """Return True if the last AI message starts with the escalation signal."""
    for msg in reversed(messages):
        role = getattr(msg, "type", getattr(msg, "role", ""))
        if role in {"ai", "assistant"}:
            content = str(getattr(msg, "content", ""))
            return _ESCALATION_SIGNAL in content
    return False


def _with_langfuse(config: RunnableConfig) -> RunnableConfig:
    """Inject a fresh Langfuse CallbackHandler into the invocation config.

    Silently skips if Langfuse is not configured (no keys in .env).
    """
    try:
        handler = get_langfuse_handler()
        callbacks = list(config.get("callbacks") or [])
        callbacks.append(handler)
        return {**config, "callbacks": callbacks}
    except Exception:
        return config


def _delta_messages(before: list, after: list) -> list:
    """Return only the messages that the agent *added* (i.e. after[len(before):])."""
    if len(after) <= len(before):
        return []
    return after[len(before):]


# ---------------------------------------------------------------------------
# Graph-node functions (shared across sessions)
# ---------------------------------------------------------------------------

def router_node(state: SupportState, config: RunnableConfig) -> dict:
    """Classify the latest user message → update state.classification."""
    from agents.router import classify_message  # lazy import
    try:
        result = classify_message(state["messages"], config=config)
        classification = result.model_dump()
    except Exception:
        classification = {"category": "general", "urgency": "low", "language": "uk"}
    return {"classification": classification}


def docs_node(state: SupportState, config: RunnableConfig) -> dict:
    """Docs Agent — customer data lookup + internal KB search."""
    from agents.docs_agent import get_docs_agent  # lazy import
    agent = get_docs_agent()
    result = agent.invoke({"messages": state["messages"]}, config=config)
    return {"messages": _delta_messages(state["messages"], result["messages"])}


def websearch_node(state: SupportState, config: RunnableConfig) -> dict:
    """Web Search Agent — operator website + DuckDuckGo."""
    from agents.websearch_agent import get_websearch_agent  # lazy import
    agent = get_websearch_agent()
    result = agent.invoke({"messages": state["messages"]}, config=config)
    return {"messages": _delta_messages(state["messages"], result["messages"])}


def off_topic_node(state: SupportState, config: RunnableConfig) -> dict:  # noqa: ARG001
    """Return a polite refusal for questions outside the Telekom support scope."""
    from langchain_core.messages import AIMessage  # lazy import
    lang = state.get("classification", {}).get("language", "uk")
    refusals = {
        "uk": (
            "Це виходить за межі моїх можливостей як агента підтримки Telekom. "
            "Я можу допомогти з питань щодо ваших послуг зв'язку, рахунків, тарифів або технічних проблем. "
            "Чим можу вам допомогти?"
        ),
        "en": (
            "That falls outside my scope as a Telekom support agent. "
            "I can help with your telecom services, billing, tariffs, or technical issues. "
            "How can I assist you?"
        ),
        "de": (
            "Das liegt außerhalb meines Aufgabenbereichs als Telekom-Support-Agent. "
            "Ich kann Ihnen bei Fragen zu Ihren Telekommunikationsdiensten, Rechnungen, Tarifen oder technischen Problemen helfen. "
            "Wie kann ich Ihnen behilflich sein?"
        ),
    }
    text = refusals.get(lang, refusals["uk"])
    return {"messages": [AIMessage(content=text)]}


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_from_router(
    state: SupportState,
) -> Literal["docs", "websearch", "escalation", "off_topic"]:
    """Pick the next agent based on Router classification."""
    category = state.get("classification", {}).get("category", "general")
    return {"product": "docs", "general": "websearch", "critical": "escalation", "off_topic": "off_topic"}.get(
        category, "websearch"
    )  # type: ignore[return-value]


def route_after_agent(state: SupportState) -> Literal["escalation", "__end__"]:
    """Route to escalation if agent emitted NEEDS_ESCALATION, otherwise END."""
    if _needs_escalation(state["messages"]):
        return "escalation"
    return END


# ---------------------------------------------------------------------------
# Graph builder (called once per session)
# ---------------------------------------------------------------------------

def build_support_graph(session_id: str = "unknown", db_path: str | None = None):
    """Build and compile a SupportGraph for one chat session.

    Args:
        session_id: Unique ID for this session. Baked into the Escalation Agent's
                    system prompt so it can pass it to send_escalation_email.
        db_path:    Path to a SQLite file for persistent checkpointing (multi-user).
                    When None (default / REPL mode) → InMemorySaver is used.
                    When provided → SqliteSaver is used so state survives restarts
                    and multiple parallel sessions share the same DB.

    Returns:
        A compiled LangGraph that accepts:
          {"messages": [...], "session_id": str, "classification": dict}
    """
    # Build session-aware escalation agent (needs session_id in system prompt)
    from agents.escalation_agent import build_escalation_agent  # lazy import
    _esc_agent = build_escalation_agent(session_id)

    def escalation_node(state: SupportState, config: RunnableConfig) -> dict:
        # Inject db_session_id into configurable so send_escalation_email can
        # read the authoritative session_id via InjectedToolArg — no globals needed.
        db_session_id = state.get("session_id", "unknown")
        esc_config = {
            **config,
            "configurable": {
                **(config.get("configurable") or {}),
                "db_session_id": db_session_id,
            },
        }
        result = _esc_agent.invoke({"messages": state["messages"]}, config=esc_config)
        return {"messages": _delta_messages(state["messages"], result["messages"])}

    # ── Build graph ──────────────────────────────────────────────────────────
    builder = StateGraph(SupportState)

    builder.add_node("router", router_node)
    builder.add_node("docs", docs_node)
    builder.add_node("websearch", websearch_node)
    builder.add_node("escalation", escalation_node)
    builder.add_node("off_topic", off_topic_node)

    builder.set_entry_point("router")

    builder.add_conditional_edges(
        "router",
        route_from_router,
        {"docs": "docs", "websearch": "websearch", "escalation": "escalation", "off_topic": "off_topic"},
    )
    builder.add_conditional_edges(
        "docs",
        route_after_agent,
        {"escalation": "escalation", END: END},
    )
    builder.add_conditional_edges(
        "websearch",
        route_after_agent,
        {"escalation": "escalation", END: END},
    )
    builder.add_edge("escalation", END)
    builder.add_edge("off_topic", END)

    # ── Choose checkpointer ──────────────────────────────────────────────────
    if db_path and _SQLITE_AVAILABLE:
        import sqlite3
        conn = sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = _SqliteSaver(conn)
    else:
        checkpointer = InMemorySaver()

    return builder.compile(checkpointer=checkpointer)

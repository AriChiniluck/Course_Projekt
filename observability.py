from __future__ import annotations

"""Centralized Langfuse helpers for lecture 12.

Why this file exists:
- keeps tracing setup in one place,
- keeps Python files free of hardcoded prompt text,
- makes the HT10 codebase ready for observability and later multi-user work.
"""

from functools import lru_cache

from langfuse import get_client
from langfuse.langchain import CallbackHandler


@lru_cache(maxsize=1)
def get_langfuse_client():
    """Return the shared Langfuse client configured via LANGFUSE_* env vars."""
    return get_client()


def get_langfuse_handler() -> CallbackHandler:
    """Return a fresh CallbackHandler linked to the current @observe trace (if any).

    Why NOT lru_cache:
      CallbackHandler is stateful — it tracks the active trace_id and span_id
      internally. A shared cached instance causes all agents (plan, research,
      critique, save_report) to contaminate each other's spans when they run
      sequentially, producing orphaned traces and missing tool-call entries in
      the Langfuse UI.

    Why no explicit trace_id (Langfuse v3):
      In Langfuse SDK v3, context propagation is handled automatically via
      Python's contextvars. When called inside an @observe-decorated function,
      CallbackHandler() picks up the active trace/span from the context
      automatically — passing trace_id= is no longer supported and raises
      TypeError. A plain CallbackHandler() is therefore correct for v3.
    """
    return CallbackHandler()


def load_prompt_from_langfuse(prompt_name: str, label: str, **variables) -> str:
    """Load and compile a prompt from Langfuse Prompt Management.

    The homework requirement says prompts must not be hardcoded in Python.
    Because of that, this function only fetches prompt content by name and label.
    """
    client = get_langfuse_client()
    prompt = client.get_prompt(prompt_name, label=label)
    return prompt.compile(**variables)


def infer_support_routing_metadata(user_input: str) -> dict:
    """Build future-ready routing metadata for the course project.

    For lecture 12 this is only a lightweight placeholder.
    In the future support project these values should come from a real Router.
    """
    text = str(user_input or "").lower()

    critical_markers = ["urgent", "asap", "refund", "angry", "critical", "problem", "error"]
    product_markers = ["price", "plan", "feature", "subscription", "policy", "tariff"]

    if any(marker in text for marker in critical_markers):
        category = "critical"
        urgency = "critical"
        confidence = 0.8
    elif any(marker in text for marker in product_markers):
        category = "product"
        urgency = "medium"
        confidence = 0.7
    else:
        category = "general"
        urgency = "low"
        confidence = 0.6

    return {
        "category": category,
        "urgency": urgency,
        "confidence": confidence,
    }

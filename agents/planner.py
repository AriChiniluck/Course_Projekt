from functools import lru_cache
import threading

from langchain.agents import create_agent
from langchain_core.tools import tool
from langfuse import observe
from observability import get_langfuse_handler
from schemas import ResearchPlan
from config import build_chat_model, settings, get_planner_system_prompt

_planner_lock = threading.Lock()


@lru_cache(maxsize=1)
def _build_planner_agent():
    return create_agent(
        model=build_chat_model(temperature=0.1, model=settings.planner_model),
        system_prompt=get_planner_system_prompt(),
        response_format=ResearchPlan,
    )


def get_planner_agent():
    """Return (lazily creating) the shared planner agent. Thread-safe on first initialisation."""
    with _planner_lock:
        return _build_planner_agent()


@tool
@observe(name="plan_tool")
def plan(request: str) -> str:
    """Create a structured research plan for the user's request."""
    planner_request = (
        f"{request}\n\n"
        "Important: keep all ResearchPlan text fields in the same language as the user's request."
    )
    result = get_planner_agent().invoke(
            {"messages": [{"role": "user", "content": planner_request}]},
            config={"callbacks": [get_langfuse_handler()]},
        )
    structured = result.get("structured_response")
    if isinstance(structured, ResearchPlan):
        return structured.model_dump_json(indent=2)

    messages = result.get("messages", [])
    if messages:
        return str(getattr(messages[-1], "content", ""))

    return "Planner did not return a valid plan."

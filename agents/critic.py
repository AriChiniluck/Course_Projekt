from functools import lru_cache
import threading

from langchain.agents import create_agent
from langchain_core.tools import tool
from langfuse import observe, propagate_attributes
from observability import get_langfuse_handler
from schemas import CritiqueResult
from config import build_chat_model, settings, get_critic_system_prompt

_critic_lock = threading.Lock()


@lru_cache(maxsize=1)
def _build_critic_agent():
    return create_agent(
        model=build_chat_model(temperature=0.0, model=settings.critic_model),
        system_prompt=get_critic_system_prompt(),
        response_format=CritiqueResult,
    )


def get_critic_agent():
    """Return (lazily creating) the shared critic agent. Thread-safe on first initialisation."""
    with _critic_lock:
        return _build_critic_agent()


@tool
@observe(name="critique_tool")
def critique(original_request: str, findings: str, plan: str = "") -> str:
    """Verify findings against the original request and return a structured approve/revise critique."""
    # Limit findings size to avoid unnecessary token spend — critic evaluates quality, not volume.
    MAX_FINDINGS_LEN = settings.critic_max_findings_len
    findings_text = str(findings or "").strip()
    if len(findings_text) > MAX_FINDINGS_LEN:
        findings_text = findings_text[:MAX_FINDINGS_LEN] + "\n\n[... findings truncated to fit evaluation context ...]"

    critique_request = "Original user request:\n" + original_request + "\n\n"
    if plan.strip():
        critique_request += "Research plan that was executed:\n" + plan.strip() + "\n\n"
    critique_request += (
        "Current research findings:\n"
        f"{findings_text}\n\n"
        "Important: return all explanation fields in the same language as the user's request/findings. "
        "If the user asked about a future year or future state beyond today, treat a clearly labeled forecast or scenario analysis based on current evidence as acceptable. Do not require impossible future facts; instead, check whether the answer transparently states uncertainty and uses the latest verified sources available today."
    )
    try:
        with propagate_attributes(metadata={"agent_name": "critic", "stage": "quality_review"}):
            result = get_critic_agent().invoke(
                {"messages": [{"role": "user", "content": critique_request}]},
                config={"callbacks": [get_langfuse_handler()]},
            )

        structured = result.get("structured_response")
        if isinstance(structured, CritiqueResult):
            return structured.model_dump_json(indent=2)

        messages = result.get("messages", [])
        if messages:
            return str(getattr(messages[-1], "content", ""))

    except Exception as exc:
        fallback = CritiqueResult(
            verdict="REVISE",
            is_error=True,
            is_fresh=False,
            is_complete=False,
            is_well_structured=False,
            strengths=[],
            gaps=[
                "The critic agent failed to return fully valid structured output.",
                str(exc),
            ],
            revision_requests=[
                "Re-run the verification and ensure all required CritiqueResult fields are present.",
            ],
        )
        return fallback.model_dump_json(indent=2)

    return CritiqueResult(
        verdict="REVISE",
        is_fresh=False,
        is_complete=False,
        is_well_structured=False,
        strengths=[],
        gaps=["Critic did not return a valid critique."],
        revision_requests=["Run the critique step again with clearer evidence."],
    ).model_dump_json(indent=2)

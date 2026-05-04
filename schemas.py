from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Support system — Routing pattern schemas
# ---------------------------------------------------------------------------

class ClassificationOutput(BaseModel):
    """Router output: classifies a customer message into a routing category."""

    category: Literal["product", "general", "critical", "off_topic"] = Field(
        description=(
            "product: account lookup, tariff/billing/FAQ questions; "
            "general: technical or external-service questions not in KB but related to Telekom; "
            "critical: blocked numbers, urgent complaints, tariff-change requests; "
            "off_topic: anything unrelated to Telekom services (weather, news, general knowledge, etc.)"
        )
    )
    urgency: Literal["low", "medium", "critical"] = Field(
        description="low: info request; medium: billing question; critical: blocking/urgent"
    )
    language: str = Field(
        default="uk",
        description="ISO 639-1 language code detected from the message, e.g. 'uk', 'ru', 'en'",
    )


class DocsResponse(BaseModel):
    """Docs Agent output: answer sourced from internal KB + customer data."""

    answer: str = Field(description="Response text for the customer")
    sources: list[str] = Field(
        default_factory=list,
        description="KB sections or file names used to compose the answer",
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence score; below 0.5 triggers escalation fallback",
    )


class WebSearchResponse(BaseModel):
    """Web Search Agent output: answer sourced from live web search."""

    answer: str = Field(description="Response text for the customer")
    sources: list[str] = Field(
        default_factory=list,
        description="URLs of web pages used",
    )
    confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence score; below 0.5 triggers escalation fallback",
    )


class EscalationOutput(BaseModel):
    """Escalation Agent output: context package sent to a human operator."""

    summary: str = Field(description="3–5 sentence summary of the issue")
    category: Literal["support", "sales"] = Field(
        description="support: technical issues/complaints; sales: tariff changes/new services"
    )
    customer_message: str = Field(description="Original customer message that triggered escalation")
    attempted_resolution: str = Field(
        description="What was tried before escalating (e.g. KB search returned no results)"
    )


class ResearchPlan(BaseModel):
    goal: str = Field(description="What we are trying to answer")
    search_queries: list[str] = Field(description="Specific queries to execute")
    sources_to_check: list[str] = Field(
        description="Use 'knowledge_base', 'web', or both"
    )
    output_format: str = Field(
        description="What the final report should look like"
    )


class CritiqueResult(BaseModel):
    verdict: Literal["APPROVE", "REVISE"] = Field(
        default="REVISE",
        description="Final reviewer decision"
    )
    is_fresh: bool = Field(
        default=False,
        description="Is the data up-to-date and based on recent sources?"
    )
    is_complete: bool = Field(
        default=False,
        description="Does the research fully cover the user's original request?"
    )
    is_well_structured: bool = Field(
        default=False,
        description="Are findings logically organized and ready for a report?"
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="What is good about the research"
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="What is missing, outdated, or poorly structured"
    )
    revision_requests: list[str] = Field(
        default_factory=list,
        description="Specific things to fix if verdict is REVISE"
    )
    is_error: bool = Field(
        default=False,
        description="True when the critique step failed due to an exception (not a quality verdict). Supervisor must not count this as a revise cycle."
    )

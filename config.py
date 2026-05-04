from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from pydantic import Field, SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_openai import ChatOpenAI

from observability import load_prompt_from_langfuse

BASE_DIR = Path(__file__).resolve().parent
TODAY = date.today().isoformat()


def _has_non_ascii_chars(path: Path) -> bool:
    try:
        str(path).encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def _get_safe_faiss_index_dir() -> Path:
    base = Path(os.getenv("LOCALAPPDATA", str(Path.home())))
    return (base / "HT_Lektion_8_FAISS" / "index").resolve()


class Settings(BaseSettings):
    openai_api_key: SecretStr = Field(..., description="OpenAI API key")
    model_name: str = Field(default="gpt-4o", description="LLM model name (fallback for all agents)")
    supervisor_model: str = Field(default="gpt-4o", description="Model for Supervisor agent")
    planner_model: str = Field(default="gpt-4o-mini", description="Model for Planner agent")
    researcher_model: str = Field(default="gpt-4o", description="Model for Researcher agent")
    critic_model: str = Field(default="gpt-4o-mini", description="Model for Critic agent")
    eval_model: str = Field(default="gpt-4o-mini", description="Model for DeepEval judges (evaluation only)")

    # --- Langfuse observability + prompt management ---
    # These values are used both for tracing and for loading prompts by name.
    langfuse_public_key: SecretStr | None = Field(default=None)
    langfuse_secret_key: SecretStr | None = Field(default=None)
    langfuse_base_url: str = Field(default="https://cloud.langfuse.com")
    langfuse_prompt_label: str = Field(default="production")
    langfuse_default_user_id: str = Field(default="student_demo")
    langfuse_project_name: str = Field(default="lecture-12")
    langfuse_project_id: str = Field(default="cmo5s4vwa014fad07fdo9jcv6")
    langfuse_org_name: str = Field(default="Test and trial.")
    langfuse_org_id: str = Field(default="cmo5s3tsj010bad08i7ijqzmk")
    langfuse_cloud_region: str = Field(default="EU")
    planner_prompt_name: str = Field(default="planner_system")
    researcher_prompt_name: str = Field(default="researcher_system")
    critic_prompt_name: str = Field(default="critic_system")
    supervisor_prompt_name: str = Field(default="supervisor_system")

    output_dir: str = Field(default="output")
    data_dir: str = Field(default="data")
    index_dir: str = Field(default="index")
    chunks_path: str = Field(default="index/chunks.json")

    embedding_model: str = Field(default="text-embedding-3-small")
    reranker_model: str = Field(default="BAAI/bge-reranker-base")

    max_search_results: int = Field(default=4, ge=1, le=20)
    max_url_content_length: int = Field(default=3000, ge=500, le=20000)
    url_fetch_timeout_sec: int = Field(default=10, ge=3, le=60)

    llm_timeout_sec: int = Field(default=90, ge=10, le=300)
    llm_max_retries: int = Field(default=2, ge=0, le=10)
    graph_recursion_limit: int = Field(default=40, ge=5, le=100)

    chunk_size: int = Field(default=800, ge=200, le=4000)
    chunk_overlap: int = Field(default=120, ge=0, le=1000)
    semantic_top_k: int = Field(default=8, ge=1, le=20)
    bm25_top_k: int = Field(default=8, ge=1, le=20)
    hybrid_top_k: int = Field(default=10, ge=1, le=20)
    rerank_top_n: int = Field(default=4, ge=1, le=10)

    critique_max_rounds: int = Field(default=3, ge=1, le=5)
    critic_max_findings_len: int = Field(default=8000, ge=500, le=50000, description="Maximum findings length (chars) passed to Critic agent")
    auto_save_min_content_len: int = Field(default=120, ge=20, le=2000, description="Min content length for Fix-B auto-save injection in supervisor middleware")
    debug: bool = Field(default=False)

    # --- API rate limiting ---
    rate_limit_sessions: str = Field(default="30/minute", description="slowapi limit for POST /sessions")
    rate_limit_chat: str = Field(default="30/minute", description="slowapi limit for POST /sessions/{id}/chat")

    # --- E-Mail escalation (SMTP) ---
    escalation_smtp_host: str = Field(default="smtp.gmail.com")
    escalation_smtp_port: int = Field(default=587)
    escalation_smtp_sender: str = Field(default="")
    escalation_smtp_password: SecretStr = Field(default=SecretStr(""))
    escalation_email_support: str = Field(default="")
    escalation_email_sales: str = Field(default="")

    # --- Operator tariff archive URL (kept in .env, not hardcoded) ---
    telekom_tariff_url: str = Field(default="")

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key(cls, value: SecretStr) -> SecretStr:
        key = value.get_secret_value().strip()
        if not key.startswith("sk-"):
            raise ValueError("OpenAI API key must start with 'sk-'.")
        if len(key) < 40:
            raise ValueError("OpenAI API key looks too short.")
        return value

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("Model name is too short.")
        return value

    @field_validator("output_dir", "data_dir", "index_dir", "chunks_path")
    @classmethod
    def resolve_project_paths(cls, value: str, info: ValidationInfo) -> str:
        path = Path(value)
        if not path.is_absolute():
            path = BASE_DIR / path
        path = path.resolve()

        if (
            info.field_name == "index_dir"
            and os.name == "nt"
            and _has_non_ascii_chars(path)
        ):
            safe_path = _get_safe_faiss_index_dir()
            safe_path.mkdir(parents=True, exist_ok=True)
            return str(safe_path)

        return str(path)


ENV_PATH = BASE_DIR / ".env"
if not ENV_PATH.exists():
    raise FileNotFoundError(
        f".env file not found at {ENV_PATH}. Copy .env.example to .env and add your OpenAI key."
    )

settings = Settings()


def _export_langfuse_env_from_settings() -> None:
    """Make sure the Langfuse SDK sees the same values as Pydantic settings.

    Langfuse usually reads LANGFUSE_* directly from the environment.
    Because this project stores them in `.env`, we mirror them into os.environ here.
    """
    if settings.langfuse_public_key:
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key.get_secret_value())
    if settings.langfuse_secret_key:
        os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key.get_secret_value())
    if settings.langfuse_base_url:
        os.environ.setdefault("LANGFUSE_BASE_URL", settings.langfuse_base_url)


_export_langfuse_env_from_settings()


def build_chat_model(temperature: float = 0.2, model: str | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or settings.model_name,
        api_key=settings.openai_api_key.get_secret_value(),
        temperature=temperature,
        timeout=settings.llm_timeout_sec,
        max_retries=settings.llm_max_retries,
    )


def _load_langfuse_prompt(prompt_name: str, **variables) -> str:
    """Load a prompt from Langfuse Prompt Management by name and label.

    If the prompt does not exist yet, we fail with a clear message so the student
    can create it in Langfuse UI using the helper markdown file in the project.
    """
    try:
        return load_prompt_from_langfuse(
            prompt_name=prompt_name,
            label=settings.langfuse_prompt_label,
            **variables,
        )
    except Exception as exc:
        raise RuntimeError(
            "Langfuse prompt loading failed. Create the required prompts in Langfuse UI "
            "with label 'production' using PROMPTS_FOR_LANGFUSE.md, then run the app again. "
            f"Missing or invalid prompt: {prompt_name}."
        ) from exc


# Lecture 12 requirement: prompts are NOT hardcoded in Python anymore.
# We fetch them lazily from Langfuse by name + label when an agent is built.
def get_planner_system_prompt() -> str:
    return _load_langfuse_prompt(settings.planner_prompt_name, today=TODAY)


def get_research_system_prompt() -> str:
    return _load_langfuse_prompt(settings.researcher_prompt_name, today=TODAY)


def get_critic_system_prompt() -> str:
    return _load_langfuse_prompt(settings.critic_prompt_name, today=TODAY)


def get_supervisor_system_prompt() -> str:
    return _load_langfuse_prompt(
        settings.supervisor_prompt_name,
        today=TODAY,
        critique_max_rounds=settings.critique_max_rounds,
    )

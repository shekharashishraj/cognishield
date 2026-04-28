from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from cognishield.app.schemas import AnswerQualityLevel, ConcernLevel

PipelineKind = Literal["legacy", "meta"]


class Settings(BaseSettings):
    """Runtime configuration: env `COGNISHIELD_*` with optional `.env` file."""

    model_config = SettingsConfigDict(
        env_prefix="COGNISHIELD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    model: str = Field(default="gpt-4o", description="OpenAI model id or vLLM served model id.")
    openai_api_base: Optional[str] = Field(
        default=None,
        description="OpenAI-compatible API base URL (e.g. http://localhost:8000/v1 for vLLM).",
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="API key; if unset, OPENAI_API_KEY from the environment is used.",
    )

    temperature_planner: float = 0.0
    temperature_generator: float = 0.3
    temperature_validators: float = 0.0
    temperature_meta: float = Field(default=0.0, description="Meta-agent structured classification.")
    temperature_revision: float = Field(
        default=0.3,
        description="Final revision pass that may align output with meta + verifier.",
    )

    pipeline: PipelineKind = Field(
        default="legacy",
        description='Orchestration: "legacy" (planner+validators loop) or "meta" (primary→meta→revision).',
    )

    meta_verifier_max_cognitive_concern: ConcernLevel = Field(
        default="medium",
        description="Reject if cognitive concern level is strictly worse than this (low < medium < high).",
    )
    meta_verifier_max_safety_concern: ConcernLevel = Field(
        default="medium",
        description="Reject if safety concern level is strictly worse than this.",
    )
    meta_verifier_min_answer_quality: AnswerQualityLevel = Field(
        default="partial",
        description="Reject if answer-direction quality is below this (inaccurate < partial < accurate).",
    )

    max_revisions: int = Field(default=3, ge=1, le=10)

    enable_bloom: bool = True
    enable_cognitive: bool = True
    enable_safety: bool = True
    enable_accuracy: bool = True

    verifier_accuracy_min_score: int = Field(default=4, ge=1, le=5)
    verifier_cognitive_min_score: int = Field(default=3, ge=1, le=5)
    verifier_bloom_min_score: int = Field(default=3, ge=1, le=5)

    log_level: str = Field(default="INFO")
    log_file: Optional[str] = Field(default=None, description="Optional path for file logging.")

    trace_path: Optional[str] = Field(
        default=None,
        description="Append JSONL trace events to this path.",
    )
    trace_stdout: bool = Field(
        default=False,
        description="If true, emit JSONL trace lines to stderr (in addition to trace_path when set).",
    )
    trace_redact: bool = Field(
        default=False,
        description="If true, omit long text fields from trace payloads.",
    )

    dry_run: bool = Field(
        default=False,
        description="If true, skip LLM calls and return deterministic stub outputs.",
    )

    # CLI-only (optional env COGNISHIELD_QUERY)
    query: str = Field(default="", description="User query when invoking from CLI.")

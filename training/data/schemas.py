"""Pydantic models for a `cognibench.jsonl` record + the paper's spec extensions.

The existing data pipeline (cognibench_pipeline.py) emits a small schema. The
paper's full spec adds age_band / persona / KORA fields. All paper extensions
are `Optional` so training works on either schema; rewards gracefully degrade
when fields are absent (see training/rewards/).
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


TurnRole = Literal["student", "teacher", "tutor", "assistant", "user"]


class Turn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: TurnRole
    content: str
    turn_number: Optional[int] = None

    def is_tutor(self) -> bool:
        return self.role in ("teacher", "tutor", "assistant")


class Conversation(BaseModel):
    """One judge-accepted dialog from `cognibench.jsonl`.

    Required: turns, subject. Everything else is optional — paper extensions
    plug in once the data pipeline emits them.
    """

    model_config = ConfigDict(extra="allow")

    # --- always present in current cognibench.jsonl ---
    turns: List[Turn]
    subject: Optional[str] = None
    problem: Optional[str] = None
    solution: Optional[str] = None
    expected_behavior: Optional[str] = None
    coercion_level: Optional[str] = None  # "none" | "moderate" | "high"
    split: Optional[str] = None
    split_label: Optional[str] = None
    metadata: dict = Field(default_factory=dict)

    # --- paper extensions (Optional; gracefully absent today) ---
    example_id: Optional[str] = None
    conversation_id: Optional[str] = None
    scenario: Optional[str] = None
    difficulty: Optional[str] = None
    policy: Optional[str] = None
    age_band: Optional[str] = None  # "7-9" | "10-12" | "13-17" | "adult"
    kora_risk_category: Optional[str] = None
    kora_motivation: Optional[str] = None
    coercion_strategy: Optional[str] = None
    student_persona: Optional[Any] = None
    turn_range: Optional[List[int]] = None
    guidance: Optional[str] = None
    tags: Optional[List[str]] = None
    judge_accepted: Optional[bool] = None  # set by future data pipeline

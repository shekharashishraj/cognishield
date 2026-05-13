"""Pydantic models for SFT dataset records.

The current data pipeline emits records in OpenAI chat format:

    {
      "conversation_id": "mt_dg_0001",
      "split": "exemplary_legitimate" | "adequate_ambiguous" | "failing_disallowed",
      "messages": [
          {"role": "system", "content": "..."},
          {"role": "user", "content": "..."},
          {"role": "assistant", "content": "..."},
          ...
      ]
    }

Paper-spec extensions (age_band, persona, kora_*) are all Optional and
gracefully absent on today's dataset — rewards fall back to safe defaults.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


MessageRole = Literal["system", "user", "assistant"]


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: MessageRole
    content: str

    def is_assistant(self) -> bool:
        return self.role == "assistant"


class Conversation(BaseModel):
    """One conversation from `sft.generated.batch.jsonl`.

    Required: `messages`. Everything else is optional metadata.
    """

    model_config = ConfigDict(extra="allow")

    messages: List[Message]
    conversation_id: Optional[str] = None
    split: Optional[str] = None  # exemplary_legitimate | adequate_ambiguous | failing_disallowed

    # --- optional metadata when the pipeline emits it ---
    subject: Optional[str] = None
    problem: Optional[str] = None
    solution: Optional[str] = None
    expected_behavior: Optional[str] = None
    coercion_level: Optional[str] = None  # "none" | "moderate" | "high"

    # --- paper-spec extensions (KORA, persona, etc.) ---
    example_id: Optional[str] = None
    scenario: Optional[str] = None
    difficulty: Optional[str] = None
    policy: Optional[str] = None
    age_band: Optional[str] = None
    kora_risk_category: Optional[str] = None
    kora_motivation: Optional[str] = None
    coercion_strategy: Optional[str] = None
    student_persona: Optional[Any] = None
    turn_range: Optional[List[int]] = None
    guidance: Optional[str] = None
    tags: Optional[List[str]] = None
    judge_accepted: Optional[bool] = None
    metadata: dict = Field(default_factory=dict)

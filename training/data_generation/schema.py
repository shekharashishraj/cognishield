from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from training.data_generation.taxonomy import (
    ANSWER_POLICIES,
    COERCION_LEVELS,
    DIFFICULTY_TO_METADATA,
    EXPECTED_BEHAVIORS,
    MATH_TOPICS,
    SCENARIOS,
    SPLITS,
    TASK_DOMAINS,
)


class RunConfig(BaseModel):
    name: str
    total_examples: int = Field(gt=0)
    max_candidate_examples: int | None = Field(default=None, gt=0)
    seed: int = 42
    output_dir: Path

    @model_validator(mode="after")
    def set_candidate_limit(self) -> "RunConfig":
        if self.max_candidate_examples is None:
            self.max_candidate_examples = self.total_examples * 3
        if self.max_candidate_examples < self.total_examples:
            raise ValueError("max_candidate_examples cannot be less than total_examples")
        return self


class GeneratorConfig(BaseModel):
    provider: Literal["openai"] = "openai"
    model: str
    temperature: float = Field(ge=0.0, le=2.0)
    max_retries: int = Field(default=3, ge=0)


class JudgeConfig(BaseModel):
    provider: Literal["openai"] = "openai"
    model: str
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class FeedbackConfig(BaseModel):
    enabled: bool = True
    max_regeneration_attempts: int = Field(default=2, ge=0)


class TurnsConfig(BaseModel):
    min_total_turns: int = Field(ge=2)
    max_total_turns: int = Field(ge=2)

    @model_validator(mode="after")
    def validate_range(self) -> "TurnsConfig":
        if self.min_total_turns > self.max_total_turns:
            raise ValueError("min_total_turns cannot exceed max_total_turns")
        if self.min_total_turns % 2 != 0 or self.max_total_turns % 2 != 0:
            raise ValueError("turn totals must be even user/assistant pairs")
        return self


class ValidationConfig(BaseModel):
    reject_schema_errors: bool = True
    reject_answer_leakage: bool = True
    reject_math_errors: bool = True
    reject_duplicate_near_matches: bool = True
    reject_first_turn_missing_problem: bool = True


class DataGenerationConfig(BaseModel):
    run: RunConfig
    generator: GeneratorConfig
    judge: JudgeConfig
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)
    difficulty_mix: dict[str, int]
    domain_mix: dict[str, int]
    scenario_mix: dict[str, int]
    policy_mix: dict[str, int]
    turns: TurnsConfig
    validation: ValidationConfig

    @field_validator("difficulty_mix")
    @classmethod
    def validate_difficulty_keys(cls, value: dict[str, int]) -> dict[str, int]:
        unknown = set(value) - set(DIFFICULTY_TO_METADATA)
        if unknown:
            raise ValueError(f"unknown difficulty keys: {sorted(unknown)}")
        return _validate_count_map(value, "difficulty_mix")

    @field_validator("domain_mix")
    @classmethod
    def validate_domain_keys(cls, value: dict[str, int]) -> dict[str, int]:
        unknown = set(value) - set(TASK_DOMAINS)
        if unknown:
            raise ValueError(f"unknown domain_mix keys: {sorted(unknown)}")
        return _validate_count_map(value, "domain_mix")

    @field_validator("scenario_mix")
    @classmethod
    def validate_scenario_keys(cls, value: dict[str, int]) -> dict[str, int]:
        unknown = set(value) - set(SCENARIOS)
        if unknown:
            raise ValueError(f"unknown scenario keys: {sorted(unknown)}")
        return _validate_count_map(value, "scenario_mix")

    @field_validator("policy_mix")
    @classmethod
    def validate_policy_keys(cls, value: dict[str, int]) -> dict[str, int]:
        unknown = set(value) - ANSWER_POLICIES
        if unknown:
            raise ValueError(f"unknown policy keys: {sorted(unknown)}")
        return _validate_count_map(value, "policy_mix")

    @model_validator(mode="after")
    def validate_totals(self) -> "DataGenerationConfig":
        total = self.run.total_examples
        for name, mix in [
            ("difficulty_mix", self.difficulty_mix),
            ("domain_mix", self.domain_mix),
            ("scenario_mix", self.scenario_mix),
            ("policy_mix", self.policy_mix),
        ]:
            if sum(mix.values()) != total:
                raise ValueError(f"{name} must sum to total_examples={total}")
        return self


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    loss_mask: bool


class Metadata(BaseModel):
    subject: str
    topic: str
    difficulty: str
    language: str = "en"
    source: str
    tags: list[str]


class Annotator(BaseModel):
    annotator_id: str
    review_status: Literal["draft", "approved"]
    notes: str


class TurnContext(BaseModel):
    learner_profile: dict[str, Any]
    rubric_constraints: dict[str, Any]
    task_context: dict[str, Any]


class GeneratedConversation(BaseModel):
    conversation_id: str
    schema_version: str
    split: str
    expected_behavior: str
    coercion_level: str
    metadata: Metadata
    annotator: Annotator
    turn_context: TurnContext
    system_prompt: dict[str, str]
    messages: list[Message]

    @model_validator(mode="after")
    def validate_taxonomy(self) -> "GeneratedConversation":
        if self.split not in SPLITS:
            raise ValueError(f"invalid split: {self.split}")
        if self.expected_behavior not in EXPECTED_BEHAVIORS:
            raise ValueError(f"invalid expected_behavior: {self.expected_behavior}")
        if self.coercion_level not in COERCION_LEVELS:
            raise ValueError(f"invalid coercion_level: {self.coercion_level}")
        policy = self.turn_context.rubric_constraints.get("tutor_answer_policy")
        if policy not in ANSWER_POLICIES:
            raise ValueError(f"invalid tutor_answer_policy: {policy}")
        if "prompt_id" not in self.system_prompt:
            raise ValueError("system_prompt.prompt_id is required")
        for key in self.turn_context.learner_profile:
            if key != "level":
                raise ValueError(f"learner_profile may only contain level, got {key}")
        if not self.messages:
            raise ValueError("messages cannot be empty")
        for message in self.messages:
            if message.role == "user" and message.loss_mask is not False:
                raise ValueError("user messages must have loss_mask=false")
            if message.role == "assistant" and message.loss_mask is not True:
                raise ValueError("assistant messages must have loss_mask=true")
        return self


def _validate_count_map(value: dict[str, int], name: str) -> dict[str, int]:
    if not value:
        raise ValueError(f"{name} cannot be empty")
    negatives = {k: v for k, v in value.items() if v < 0}
    if negatives:
        raise ValueError(f"{name} contains negative counts: {negatives}")
    return value


def load_config(path: Path) -> DataGenerationConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return DataGenerationConfig.model_validate(raw)


def load_conversation_json(path: Path) -> GeneratedConversation:
    with path.open("r", encoding="utf-8") as f:
        return GeneratedConversation.model_validate(json.load(f))


def conversation_to_json(conversation: GeneratedConversation | dict[str, Any]) -> str:
    if isinstance(conversation, GeneratedConversation):
        payload = conversation.model_dump(mode="json")
    else:
        payload = conversation
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

InterventionType = Literal["scaffold", "hint", "redirect", "defer", "refuse"]


class PlannerOutput(BaseModel):
    intervention: InterventionType
    policy_rationale: str
    generator_instruction: str


class GeneratorOutput(BaseModel):
    response_text: str
    self_check: str


class ValidatorOutput(BaseModel):
    validator_name: str
    score: int = Field(ge=1, le=5)
    passed: bool
    issues: List[str] = Field(default_factory=list)
    fix_suggestion: str = "none"

    answer_leakage: Optional[bool] = None
    direct_solution: Optional[bool] = None
    prompt_injection_detected: Optional[bool] = None
    level_match: Optional[bool] = None
    does_reasoning_for_student: Optional[bool] = None
    factual_risk: Optional[bool] = None


class VerifierDecision(BaseModel):
    decision: Literal["accept", "revise"]
    reasons: List[str] = Field(default_factory=list)
    backprompt: Optional[str] = None


class TurnContext(BaseModel):
    user_query: str
    history: List[Dict[str, str]] = Field(default_factory=list)
    learner_profile: Dict[str, Any] = Field(default_factory=dict)
    rubric_constraints: Dict[str, Any] = Field(default_factory=dict)
    task_context: Dict[str, Any] = Field(default_factory=dict)


class CogniShieldState(BaseModel):
    context: TurnContext
    plan: Optional[PlannerOutput] = None
    candidate: Optional[GeneratorOutput] = None
    validator_reports: Dict[str, ValidatorOutput] = Field(default_factory=dict)
    backprompt: Optional[str] = None
    attempt: int = 0

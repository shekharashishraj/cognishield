from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

InterventionType = Literal["scaffold", "hint", "redirect", "defer", "refuse"]

# Meta pipeline: severity of concern for cognitive/safety (low = little concern / better).
ConcernLevel = Literal["low", "medium", "high"]

# Meta pipeline: factual/directional quality of the draft toward a correct solution.
AnswerQualityLevel = Literal["inaccurate", "partial", "accurate"]


class CognitiveSafetyClassifier(BaseModel):
    """Concern severity for cognitive engagement or safety risk (low is preferable)."""

    level: ConcernLevel
    reason: str


class AnswerDirectionClassifier(BaseModel):
    """Quality of hints/direction relative to a correct solution."""

    level: AnswerQualityLevel
    reason: str


class MetaAgentOutput(BaseModel):
    """Structured output from the meta-agent (one LLM call)."""

    cognitive_classifier: CognitiveSafetyClassifier
    safety_classifier: CognitiveSafetyClassifier
    answer_classifier: AnswerDirectionClassifier


class RevisionOutput(BaseModel):
    """Final user-facing response after optional alignment with meta + verifier."""

    response_text: str


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
    # Meta pipeline (primary → meta → revision); legacy pipeline leaves these unset.
    primary_draft: Optional[GeneratorOutput] = None
    meta_output: Optional[MetaAgentOutput] = None
    meta_verifier_decision: Optional[VerifierDecision] = None
    final_response_text: Optional[str] = None

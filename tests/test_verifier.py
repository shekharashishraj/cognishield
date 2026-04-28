from __future__ import annotations

from cognishield.app.schemas import (
    AnswerDirectionClassifier,
    CognitiveSafetyClassifier,
    GeneratorOutput,
    MetaAgentOutput,
    PlannerOutput,
    ValidatorOutput,
)
from cognishield.app.settings import Settings
from cognishield.app.verifier import verify_meta_classifiers, verify_with_rules


def _meta(
    *,
    cognitive: str = "low",
    safety: str = "low",
    answer: str = "accurate",
) -> MetaAgentOutput:
    return MetaAgentOutput(
        cognitive_classifier=CognitiveSafetyClassifier(level=cognitive, reason="test"),
        safety_classifier=CognitiveSafetyClassifier(level=safety, reason="test"),
        answer_classifier=AnswerDirectionClassifier(level=answer, reason="test"),
    )


def _reports(
    *,
    bloom_score: int = 5,
    cognitive_score: int = 5,
    safety_leak: bool = False,
    accuracy_score: int = 5,
) -> dict[str, ValidatorOutput]:
    return {
        "bloom": ValidatorOutput(
            validator_name="bloom",
            score=bloom_score,
            passed=bloom_score >= 3,
            issues=[],
        ),
        "cognitive": ValidatorOutput(
            validator_name="cognitive",
            score=cognitive_score,
            passed=cognitive_score >= 3,
            issues=[],
        ),
        "safety": ValidatorOutput(
            validator_name="safety",
            score=5,
            passed=True,
            issues=[],
            answer_leakage=safety_leak,
            direct_solution=False,
            prompt_injection_detected=False,
        ),
        "accuracy": ValidatorOutput(
            validator_name="accuracy",
            score=accuracy_score,
            passed=accuracy_score >= 4,
            issues=[],
        ),
    }


def test_verifier_accepts_when_all_clear() -> None:
    plan = PlannerOutput(
        intervention="hint",
        policy_rationale="ok",
        generator_instruction="hint only",
    )
    candidate = GeneratorOutput(response_text="Try X", self_check="ok")
    settings = Settings()
    verdict = verify_with_rules(plan, candidate, _reports(), settings)
    assert verdict.decision == "accept"


def test_verifier_revises_on_answer_leakage() -> None:
    plan = PlannerOutput(
        intervention="hint",
        policy_rationale="ok",
        generator_instruction="hint only",
    )
    candidate = GeneratorOutput(response_text="leaky", self_check="ok")
    settings = Settings()
    verdict = verify_with_rules(plan, candidate, _reports(safety_leak=True), settings)
    assert verdict.decision == "revise"
    assert verdict.backprompt


def test_meta_verifier_accepts_when_thresholds_pass() -> None:
    settings = Settings()
    verdict = verify_meta_classifiers(_meta(), settings)
    assert verdict.decision == "accept"


def test_meta_verifier_revises_on_high_cognitive() -> None:
    settings = Settings(meta_verifier_max_cognitive_concern="medium")
    verdict = verify_meta_classifiers(_meta(cognitive="high"), settings)
    assert verdict.decision == "revise"
    assert verdict.backprompt


def test_meta_verifier_revises_on_high_safety() -> None:
    settings = Settings(meta_verifier_max_safety_concern="low")
    verdict = verify_meta_classifiers(_meta(safety="medium"), settings)
    assert verdict.decision == "revise"


def test_meta_verifier_revises_on_inaccurate_answer() -> None:
    settings = Settings(meta_verifier_min_answer_quality="partial")
    verdict = verify_meta_classifiers(_meta(answer="inaccurate"), settings)
    assert verdict.decision == "revise"


def test_verifier_respects_custom_accuracy_threshold() -> None:
    plan = PlannerOutput(
        intervention="hint",
        policy_rationale="ok",
        generator_instruction="hint only",
    )
    candidate = GeneratorOutput(response_text="maybe wrong", self_check="unsure")
    settings = Settings(verifier_accuracy_min_score=5)
    verdict = verify_with_rules(plan, candidate, _reports(accuracy_score=4), settings)
    assert verdict.decision == "revise"

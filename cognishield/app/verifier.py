from __future__ import annotations

from cognishield.app.schemas import (
    GeneratorOutput,
    PlannerOutput,
    ValidatorOutput,
    VerifierDecision,
)
from cognishield.app.settings import Settings


def verify_with_rules(
    plan: PlannerOutput,
    candidate: GeneratorOutput,
    reports: dict[str, ValidatorOutput],
    settings: Settings,
) -> VerifierDecision:
    reasons: list[str] = []

    bloom = reports["bloom"]
    cognitive = reports["cognitive"]
    safety = reports["safety"]
    accuracy = reports["accuracy"]

    if safety.answer_leakage:
        reasons.append("Answer leakage detected.")
    if safety.direct_solution:
        reasons.append("Response gives too direct a solution.")
    if safety.prompt_injection_detected:
        reasons.append("Prompt injection or policy bypass risk detected.")
    if accuracy.score < settings.verifier_accuracy_min_score:
        reasons.append("Accuracy too low.")
    if cognitive.score < settings.verifier_cognitive_min_score:
        reasons.append("Cognitive engagement too low.")
    if bloom.score < settings.verifier_bloom_min_score:
        reasons.append("Pedagogical level mismatch.")

    if reasons:
        return VerifierDecision(
            decision="revise",
            reasons=reasons,
            backprompt=(
                "Revise the answer to remove answer-level disclosure, increase scaffolding, "
                "preserve factual correctness, and require more student reasoning."
            ),
        )

    return VerifierDecision(
        decision="accept",
        reasons=["All verifier thresholds passed."],
        backprompt=None,
    )

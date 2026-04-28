from __future__ import annotations

from cognishield.app.schemas import (
    GeneratorOutput,
    MetaAgentOutput,
    PlannerOutput,
    ValidatorOutput,
    VerifierDecision,
)
from cognishield.app.settings import Settings

_CONCERN_RANK = {"low": 0, "medium": 1, "high": 2}
_ANSWER_RANK = {"inaccurate": 0, "partial": 1, "accurate": 2}

_META_REVISE_BACKPROMPT = (
    "Revise the draft using the meta feedback: improve scaffolding, reduce direct answers "
    "where inappropriate, fix directional errors, and resolve any safety issues noted."
)


def verify_meta_classifiers(meta: MetaAgentOutput, settings: Settings) -> VerifierDecision:
    """Deterministic gate on structured meta-agent output (no extra LLM)."""

    reasons: list[str] = []

    max_cognitive = _CONCERN_RANK[settings.meta_verifier_max_cognitive_concern]
    if _CONCERN_RANK[meta.cognitive_classifier.level] > max_cognitive:
        reasons.append("Cognitive concern above threshold.")

    max_safety = _CONCERN_RANK[settings.meta_verifier_max_safety_concern]
    if _CONCERN_RANK[meta.safety_classifier.level] > max_safety:
        reasons.append("Safety concern above threshold.")

    min_answer = _ANSWER_RANK[settings.meta_verifier_min_answer_quality]
    if _ANSWER_RANK[meta.answer_classifier.level] < min_answer:
        reasons.append("Answer-direction quality below minimum.")

    if reasons:
        return VerifierDecision(
            decision="revise",
            reasons=reasons,
            backprompt=_META_REVISE_BACKPROMPT,
        )

    return VerifierDecision(
        decision="accept",
        reasons=["All meta verifier thresholds passed."],
        backprompt=None,
    )


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

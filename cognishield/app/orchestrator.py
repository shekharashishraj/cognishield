from __future__ import annotations

import json
from typing import Any, Callable, Optional

from cognishield.app.chains.generator_chain import build_generator_chain
from cognishield.app.chains.meta_chain import build_meta_chain
from cognishield.app.chains.planner_chain import build_planner_chain
from cognishield.app.chains.primary_chain import build_primary_chain
from cognishield.app.chains.revision_chain import build_revision_chain
from cognishield.app.chains.validator_chains import (
    build_accuracy_validator_chain,
    build_bloom_validator_chain,
    build_cognitive_validator_chain,
    build_safety_validator_chain,
)
from cognishield.app.logging_setup import get_logger
from cognishield.app.schemas import (
    AnswerDirectionClassifier,
    CogniShieldState,
    CognitiveSafetyClassifier,
    GeneratorOutput,
    MetaAgentOutput,
    PlannerOutput,
    RevisionOutput,
    ValidatorOutput,
)
from cognishield.app.settings import Settings
from cognishield.app.trace import JsonlTracer
from cognishield.app.verifier import verify_meta_classifiers, verify_with_rules

_LOG = get_logger("cognishield.orchestrator")

EmitFn = Callable[[str, dict[str, Any]], None]

FALLBACK_RESPONSE = (
    "I can help you work through this step by step. "
    "Start by telling me what you've tried so far, and I'll guide you with a hint."
)


def _serialize(obj: Any) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)


def _stub_plan() -> PlannerOutput:
    return PlannerOutput(
        intervention="hint",
        policy_rationale="dry_run stub",
        generator_instruction="Offer a conceptual hint without giving the final answer.",
    )


def _stub_candidate(backprompt: str) -> GeneratorOutput:
    text = (
        "[dry_run] Here's a small nudge: identify what quantity the problem asks for, "
        "then list the givens you can use."
    )
    if backprompt:
        text += f" (revised per: {backprompt[:80]})"
    return GeneratorOutput(response_text=text, self_check="dry_run stub")


def _stub_validator(name: str) -> ValidatorOutput:
    return ValidatorOutput(
        validator_name=name,
        score=5,
        passed=True,
        issues=[],
        fix_suggestion="none",
    )


def _stub_primary_draft() -> GeneratorOutput:
    return GeneratorOutput(
        response_text=(
            "[dry_run] Primary: What quantity are you solving for, and what values are given?"
        ),
        self_check="dry_run stub",
    )


def _stub_meta_output() -> MetaAgentOutput:
    return MetaAgentOutput(
        cognitive_classifier=CognitiveSafetyClassifier(
            level="low",
            reason="dry_run: scaffolding leaves reasoning to the student.",
        ),
        safety_classifier=CognitiveSafetyClassifier(
            level="low",
            reason="dry_run: no harmful or policy-bypass content.",
        ),
        answer_classifier=AnswerDirectionClassifier(
            level="accurate",
            reason="dry_run: hint points toward the correct approach.",
        ),
    )


def _stub_revision_text() -> str:
    return (
        "[dry_run] Final: Start by naming the target quantity, then list givens and constraints."
    )


def _context_payload(state: CogniShieldState) -> dict[str, str]:
    ctx = state.context
    return {
        "user_query": ctx.user_query,
        "history": _serialize(ctx.history),
        "learner_profile": _serialize(ctx.learner_profile),
        "rubric_constraints": _serialize(ctx.rubric_constraints),
        "task_context": _serialize(ctx.task_context),
    }


def _dry_run_legacy_turn(state: CogniShieldState, settings: Settings, emit: EmitFn) -> str:
    state.plan = _stub_plan()
    emit("plan", {"dry_run": True, "plan": state.plan.model_dump()})
    backprompt = ""
    for attempt in range(settings.max_revisions):
        state.attempt = attempt + 1
        state.candidate = _stub_candidate(backprompt)
        emit(
            "generate",
            {
                "dry_run": True,
                "attempt": state.attempt,
                "candidate": state.candidate.model_dump(),
            },
        )
        reports = {
            "bloom": _stub_validator("bloom"),
            "cognitive": _stub_validator("cognitive"),
            "safety": _stub_validator("safety"),
            "accuracy": _stub_validator("accuracy"),
        }
        state.validator_reports = reports
        for name, rep in reports.items():
            emit(f"validate:{name}", {"dry_run": True, "report": rep.model_dump()})
        verdict = verify_with_rules(state.plan, state.candidate, reports, settings)
        emit("verify", verdict.model_dump())
        if verdict.decision == "accept":
            _LOG.info("accepted after %s attempts (dry_run)", state.attempt)
            return state.candidate.response_text
        backprompt = verdict.backprompt or ""
        emit("revise", {"attempt": state.attempt, "backprompt": backprompt})
    return FALLBACK_RESPONSE


def _dry_run_meta_turn(state: CogniShieldState, settings: Settings, emit: EmitFn) -> str:
    state.primary_draft = _stub_primary_draft()
    emit("primary", {"dry_run": True, "draft": state.primary_draft.model_dump()})
    state.meta_output = _stub_meta_output()
    emit("meta", {"dry_run": True, "meta": state.meta_output.model_dump()})
    verdict = verify_meta_classifiers(state.meta_output, settings)
    state.meta_verifier_decision = verdict
    emit("verify", verdict.model_dump())
    final = RevisionOutput(response_text=_stub_revision_text())
    state.final_response_text = final.response_text
    emit("revision", {"dry_run": True, "final": final.model_dump()})
    return final.response_text


def _run_legacy_turn(state: CogniShieldState, settings: Settings, emit: EmitFn) -> str:
    ctx = state.context
    planner_chain = build_planner_chain(settings)
    generator_chain = build_generator_chain(settings)
    validators: list[tuple[str, bool, Callable[[], Any]]] = [
        ("bloom", settings.enable_bloom, lambda: build_bloom_validator_chain(settings)),
        ("cognitive", settings.enable_cognitive, lambda: build_cognitive_validator_chain(settings)),
        ("safety", settings.enable_safety, lambda: build_safety_validator_chain(settings)),
        ("accuracy", settings.enable_accuracy, lambda: build_accuracy_validator_chain(settings)),
    ]

    planner_payload = {
        "user_query": ctx.user_query,
        "history": _serialize(ctx.history),
        "learner_profile": _serialize(ctx.learner_profile),
        "rubric_constraints": _serialize(ctx.rubric_constraints),
        "task_context": _serialize(ctx.task_context),
    }
    state.plan = planner_chain.invoke(planner_payload)
    emit("plan", {"plan": state.plan.model_dump()})
    _LOG.info("Planner intervention=%s", state.plan.intervention)

    previous_candidate = ""
    backprompt = ""

    for attempt in range(settings.max_revisions):
        state.attempt = attempt + 1
        gen_payload = {
            "user_query": ctx.user_query,
            "history": _serialize(ctx.history),
            "intervention": state.plan.intervention,
            "policy_rationale": state.plan.policy_rationale,
            "generator_instruction": state.plan.generator_instruction,
            "previous_candidate": previous_candidate,
            "backprompt": backprompt,
        }
        state.candidate = generator_chain.invoke(gen_payload)
        emit(
            "generate",
            {"attempt": state.attempt, "candidate": state.candidate.model_dump()},
        )
        _LOG.info("Generated candidate len=%s", len(state.candidate.response_text))

        validator_payload = {
            "user_query": ctx.user_query,
            "history": _serialize(ctx.history),
            "intervention": state.plan.intervention,
            "generator_instruction": state.plan.generator_instruction,
            "candidate_response": state.candidate.response_text,
            "rubric_constraints": _serialize(ctx.rubric_constraints),
            "task_context": _serialize(ctx.task_context),
        }

        reports: dict[str, ValidatorOutput] = {}
        for name, enabled, builder in validators:
            if not enabled:
                reports[name] = ValidatorOutput(
                    validator_name=name,
                    score=5,
                    passed=True,
                    issues=[],
                    fix_suggestion="none",
                )
                emit(
                    f"validate:{name}",
                    {"skipped": True, "report": reports[name].model_dump()},
                )
                continue
            chain = builder()
            reports[name] = chain.invoke(validator_payload)
            emit(f"validate:{name}", {"report": reports[name].model_dump()})
            _LOG.info(
                "Validator %s score=%s passed=%s",
                name,
                reports[name].score,
                reports[name].passed,
            )

        state.validator_reports = reports
        verdict = verify_with_rules(state.plan, state.candidate, reports, settings)
        emit("verify", verdict.model_dump())
        if verdict.decision == "accept":
            _LOG.info("Verifier accepted after %s attempts", state.attempt)
            emit("accept", {"attempt": state.attempt})
            return state.candidate.response_text

        previous_candidate = state.candidate.response_text
        backprompt = verdict.backprompt or ""
        emit(
            "revise",
            {"attempt": state.attempt, "reasons": verdict.reasons, "backprompt": backprompt},
        )
        _LOG.info("Verifier requested revision: %s", verdict.reasons)

    _LOG.warning("Max revisions exhausted")
    emit("max_revisions", {"max": settings.max_revisions})
    return FALLBACK_RESPONSE


def _run_meta_turn(state: CogniShieldState, settings: Settings, emit: EmitFn) -> str:
    base = _context_payload(state)
    primary_chain = build_primary_chain(settings)
    meta_chain = build_meta_chain(settings)
    revision_chain = build_revision_chain(settings)

    state.primary_draft = primary_chain.invoke(base)
    emit("primary", {"draft": state.primary_draft.model_dump()})

    meta_payload = {**base, "draft_response": state.primary_draft.response_text}
    state.meta_output = meta_chain.invoke(meta_payload)
    emit("meta", {"meta": state.meta_output.model_dump()})

    verdict = verify_meta_classifiers(state.meta_output, settings)
    state.meta_verifier_decision = verdict
    emit("verify", verdict.model_dump())

    revision_payload = {
        **base,
        "draft_response": state.primary_draft.response_text,
        "meta_json": _serialize(state.meta_output.model_dump()),
        "verifier_json": _serialize(verdict.model_dump()),
    }
    final = revision_chain.invoke(revision_payload)
    state.final_response_text = final.response_text
    emit("revision", {"final": final.model_dump()})
    _LOG.info("Meta pipeline completed; final len=%s", len(final.response_text))
    return final.response_text


def run_turn(
    state: CogniShieldState,
    settings: Settings,
    tracer: Optional[JsonlTracer] = None,
) -> str:
    trace = tracer

    def emit(stage: str, payload: dict[str, Any]) -> None:
        if trace:
            trace.event(stage, payload)
        _LOG.info("%s", stage)

    if settings.dry_run:
        if settings.pipeline == "meta":
            return _dry_run_meta_turn(state, settings, emit)
        return _dry_run_legacy_turn(state, settings, emit)

    if settings.pipeline == "meta":
        return _run_meta_turn(state, settings, emit)
    return _run_legacy_turn(state, settings, emit)

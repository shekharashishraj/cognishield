from __future__ import annotations

import json
from typing import Any, Callable, Optional

from cognishield.app.chains.generator_chain import build_generator_chain
from cognishield.app.chains.planner_chain import build_planner_chain
from cognishield.app.chains.validator_chains import (
    build_accuracy_validator_chain,
    build_bloom_validator_chain,
    build_cognitive_validator_chain,
    build_safety_validator_chain,
)
from cognishield.app.logging_setup import get_logger
from cognishield.app.schemas import (
    CogniShieldState,
    GeneratorOutput,
    PlannerOutput,
    ValidatorOutput,
)
from cognishield.app.settings import Settings
from cognishield.app.trace import JsonlTracer
from cognishield.app.verifier import verify_with_rules

_LOG = get_logger("cognishield.orchestrator")

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


def run_turn(
    state: CogniShieldState,
    settings: Settings,
    tracer: Optional[JsonlTracer] = None,
) -> str:
    ctx = state.context
    trace = tracer

    def emit(stage: str, payload: dict[str, Any]) -> None:
        if trace:
            trace.event(stage, payload)
        _LOG.info("%s", stage)

    if settings.dry_run:
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
            previous_candidate = state.candidate.response_text
            backprompt = verdict.backprompt or ""
            emit("revise", {"attempt": state.attempt, "backprompt": backprompt})
        return FALLBACK_RESPONSE

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
            _LOG.info("Validator %s score=%s passed=%s", name, reports[name].score, reports[name].passed)

        state.validator_reports = reports
        verdict = verify_with_rules(state.plan, state.candidate, reports, settings)
        emit("verify", verdict.model_dump())
        if verdict.decision == "accept":
            _LOG.info("Verifier accepted after %s attempts", state.attempt)
            emit("accept", {"attempt": state.attempt})
            return state.candidate.response_text

        previous_candidate = state.candidate.response_text
        backprompt = verdict.backprompt or ""
        emit("revise", {"attempt": state.attempt, "reasons": verdict.reasons, "backprompt": backprompt})
        _LOG.info("Verifier requested revision: %s", verdict.reasons)

    _LOG.warning("Max revisions exhausted")
    emit("max_revisions", {"max": settings.max_revisions})
    return FALLBACK_RESPONSE

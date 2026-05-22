from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.data_generation.logging_utils import RunLogger, write_summary
from training.data_generation.llm_judge import judge_conversation_with_openai
from training.data_generation.openai_client import generate_conversation_with_openai
from training.data_generation.planning import PlannedExample, build_generation_plan, save_generation_plan
from training.data_generation.schema import GeneratedConversation, conversation_to_json, load_config
from training.data_generation.validators import ValidationResult, validate_conversation


@dataclass(frozen=True)
class CandidateGenerationResult:
    accepted: bool
    regenerated: bool


_LOG_PLAN_TEXT_MAX = 800


def _planned_example_log_payload(planned: PlannedExample) -> dict[str, Any]:
    payload = asdict(planned)
    for key in ("problem_statement", "reference_solution"):
        val = payload.get(key)
        if isinstance(val, str) and len(val) > _LOG_PLAN_TEXT_MAX:
            payload[key] = val[:_LOG_PLAN_TEXT_MAX] + "...[truncated]"
    return payload


def generate(config_path: Path) -> int:
    config = load_config(config_path)
    run_dir = config.run.output_dir
    raw_dir = run_dir / "raw"
    generation_rejected_dir = run_dir / "generation_rejected"
    raw_dir.mkdir(parents=True, exist_ok=True)
    generation_rejected_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "valid").mkdir(parents=True, exist_ok=True)
    (run_dir / "rejected").mkdir(parents=True, exist_ok=True)
    for directory in (raw_dir, generation_rejected_dir, run_dir / "valid", run_dir / "rejected"):
        for stale in directory.glob("*.json"):
            stale.unlink()
    logger = RunLogger(run_dir, "training.data_generation.generate")
    logger.event(
        "run_start",
        config_path=str(config_path),
        seed=config.run.seed,
        generator_model=config.generator.model,
        judge_model=config.judge.model,
        output_dir=str(run_dir),
        total_examples=config.run.total_examples,
        max_candidate_examples=config.run.max_candidate_examples,
        feedback_enabled=config.feedback.enabled,
        max_regeneration_attempts=config.feedback.max_regeneration_attempts,
    )
    shutil.copy2(config_path, run_dir / "config.yaml")

    all_planned = build_generation_plan(config)
    save_generation_plan(all_planned, run_dir / "generation_plan.json")
    for planned in all_planned:
        logger.event("planned_example", **_planned_example_log_payload(planned))

    pending = deque(all_planned)
    target = config.run.total_examples
    max_candidate_examples = config.run.max_candidate_examples or target * 3
    candidate_examples = len(all_planned)
    generated = 0
    failed = 0
    regenerated = 0
    replacements_created = 0

    while pending and generated < target:
        planned = pending.popleft()
        candidate_result = _generate_planned_example(
            config=config,
            planned=planned,
            raw_dir=raw_dir,
            generation_rejected_dir=generation_rejected_dir,
            logger=logger,
        )
        if candidate_result.accepted:
            generated += 1
            if candidate_result.regenerated:
                regenerated += 1
            continue

        failed += 1
        if candidate_examples >= max_candidate_examples:
            logger.event(
                "candidate_exhausted_no_replacement",
                example_id=planned.example_id,
                target=target,
                generated=generated,
                candidate_examples=candidate_examples,
                max_candidate_examples=max_candidate_examples,
            )
            continue

        candidate_examples += 1
        replacements_created += 1
        replacement = _replacement_for(planned, candidate_examples)
        all_planned.append(replacement)
        pending.append(replacement)
        save_generation_plan(all_planned, run_dir / "generation_plan.json")
        logger.event(
            "planned_example",
            **_planned_example_log_payload(replacement),
            replacement_for=planned.example_id,
        )

    save_generation_plan(all_planned, run_dir / "generation_plan.json")
    summary: dict[str, Any] = {
        "summary_stage": "generation",
        "target": target,
        "generated": generated,
        "failed": failed,
        "regenerated": regenerated,
        "candidate_examples": candidate_examples,
        "max_candidate_examples": max_candidate_examples,
        "replacements_created": replacements_created,
        "target_met": generated >= target,
    }
    write_summary(run_dir, summary)
    logger.event("run_end", **summary)
    return 0 if generated >= target else 1


def _generate_planned_example(
    *,
    config,
    planned,
    raw_dir: Path,
    generation_rejected_dir: Path,
    logger: RunLogger,
) -> CandidateGenerationResult:
    feedback_issues: list[dict[str, str]] | None = None
    previous_payload: dict[str, Any] | None = None
    max_generation_attempts = (
        config.feedback.max_regeneration_attempts + 1
        if config.feedback.enabled
        else 1
    )
    for generation_attempt in range(max_generation_attempts):
        payload: dict[str, Any] | None = None
        for api_attempt in range(config.generator.max_retries + 1):
            try:
                logger.event(
                    "openai_call_start",
                    example_id=planned.example_id,
                    generation_attempt=generation_attempt + 1,
                    api_attempt=api_attempt + 1,
                    has_feedback=bool(feedback_issues),
                )
                started = time.monotonic()
                payload, meta = generate_conversation_with_openai(
                    config=config,
                    planned=planned,
                    feedback_issues=feedback_issues,
                    previous_payload=previous_payload,
                )
                latency_s = round(time.monotonic() - started, 3)
                logger.event(
                    "openai_call_success",
                    example_id=planned.example_id,
                    generation_attempt=generation_attempt + 1,
                    api_attempt=api_attempt + 1,
                    latency_s=latency_s,
                    provider_meta=meta,
                )
                break
            except Exception as exc:  # pragma: no cover - network path
                logger.error(
                    "openai_call_error",
                    exc,
                    example_id=planned.example_id,
                    generation_attempt=generation_attempt + 1,
                    api_attempt=api_attempt + 1,
                )
                if api_attempt >= config.generator.max_retries:
                    payload = None
        if payload is None:
            continue

        validation_result = _validate_generated_payload(
            config=config,
            planned=planned,
            payload=payload,
            output_path=raw_dir / planned.filename,
        )
        if validation_result.passed:
            out_path = raw_dir / planned.filename
            out_path.write_text(conversation_to_json(payload), encoding="utf-8")
            logger.event(
                "generation_accept",
                example_id=planned.example_id,
                generation_attempt=generation_attempt + 1,
                output=str(out_path),
            )
            return CandidateGenerationResult(
                accepted=True,
                regenerated=generation_attempt > 0,
            )

        issues = [issue.__dict__ for issue in validation_result.issues]
        rejected_path = generation_rejected_dir / (
            f"{planned.example_id}_attempt_{generation_attempt + 1}.json"
        )
        rejected_path.write_text(conversation_to_json(payload), encoding="utf-8")
        logger.event(
            "generation_validation_fail",
            example_id=planned.example_id,
            generation_attempt=generation_attempt + 1,
            rejected_output=str(rejected_path),
            issues=issues,
        )
        feedback_issues = issues
        previous_payload = payload

    return CandidateGenerationResult(accepted=False, regenerated=False)


def _replacement_for(planned, sequence_number: int):
    example_id = f"dg_{sequence_number:04d}"
    return replace(
        planned,
        example_id=example_id,
        filename=f"{example_id}.json",
        conversation_id=f"mt_{example_id}",
    )


def _validate_generated_payload(
    *,
    config,
    planned,
    payload: dict[str, Any],
    output_path: Path,
) -> ValidationResult:
    result = ValidationResult(passed=True)
    try:
        conversation = GeneratedConversation.model_validate(payload)
    except Exception as exc:
        result.add("schema_error", str(exc))
        result.passed = False
        return result

    result = validate_conversation(
        path=output_path,
        conversation=conversation,
        min_total_turns=config.turns.min_total_turns,
        max_total_turns=config.turns.max_total_turns,
        scenario=planned.scenario,
        reject_answer_leakage=config.validation.reject_answer_leakage,
        reject_first_turn_missing_problem=config.validation.reject_first_turn_missing_problem,
    )
    if result.passed and (
        config.validation.reject_math_errors or config.validation.reject_answer_leakage
    ):
        try:
            judge_issues = judge_conversation_with_openai(
                config=config,
                conversation=conversation,
            )
        except Exception as exc:
            judge_issues = [
                {
                    "code": "llm_judge_error",
                    "message": f"LLM judge raised an unexpected exception: {exc}",
                }
            ]
        for issue in judge_issues:
            result.add(issue["code"], issue["message"])
        result.passed = not result.issues
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic math tutoring SFT data.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    return generate(args.config)


if __name__ == "__main__":
    raise SystemExit(main())

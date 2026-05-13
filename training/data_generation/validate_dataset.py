from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pydantic import ValidationError

from training.data_generation.llm_judge import judge_conversation_with_openai
from training.data_generation.logging_utils import RunLogger, write_summary
from training.data_generation.schema import GeneratedConversation, load_config
from training.data_generation.validators import load_json_object, validate_conversation


def validate_run(
    run_dir: Path,
    config_path: Path | None = None,
    *,
    run_llm_judge: bool = True,
) -> int:
    if config_path is None:
        config_path = run_dir / "config.yaml"
    config = load_config(config_path)
    raw_dir = run_dir / "raw"
    valid_dir = run_dir / "valid"
    rejected_dir = run_dir / "rejected"
    valid_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)
    for directory in (valid_dir, rejected_dir):
        for stale in directory.glob("*.json"):
            stale.unlink()
    logger = RunLogger(run_dir, "training.data_generation.validate")
    logger.event("validation_start", run_dir=str(run_dir), config_path=str(config_path))

    plan_by_filename = {
        item["filename"]: item
        for item in json.loads((run_dir / "generation_plan.json").read_text(encoding="utf-8"))
    }
    accepted = 0
    rejected = 0
    counters: dict[str, Counter[str]] = {
        "scenario": Counter(),
        "difficulty": Counter(),
        "policy": Counter(),
        "coercion": Counter(),
        "subject": Counter(),
    }

    for path in sorted(raw_dir.glob("*.json")):
        plan_item = plan_by_filename.get(path.name, {})
        try:
            raw = load_json_object(path)
            conversation = GeneratedConversation.model_validate(raw)
            result = validate_conversation(
                path=path,
                conversation=conversation,
                min_total_turns=config.turns.min_total_turns,
                max_total_turns=config.turns.max_total_turns,
                scenario=plan_item.get("scenario"),
                reject_answer_leakage=config.validation.reject_answer_leakage,
                reject_first_turn_missing_problem=config.validation.reject_first_turn_missing_problem,
            )
            if run_llm_judge and result.passed and (
                config.validation.reject_math_errors or config.validation.reject_answer_leakage
            ):
                judge_issues = judge_conversation_with_openai(
                    config=config,
                    conversation=conversation,
                )
                for issue in judge_issues:
                    result.add(issue["code"], issue["message"])
                result.passed = not result.issues
        except (json.JSONDecodeError, OSError, ValidationError, ValueError) as exc:
            logger.error("validation_exception", exc, example_id=path.stem, input=str(path))
            result = None

        if result and result.passed:
            shutil.copy2(path, valid_dir / path.name)
            accepted += 1
            counters["scenario"][plan_item.get("scenario", "unknown")] += 1
            counters["difficulty"][plan_item.get("difficulty_level", "unknown")] += 1
            counters["policy"][conversation.turn_context.rubric_constraints.get("tutor_answer_policy", "unknown")] += 1
            counters["coercion"][conversation.coercion_level] += 1
            counters["subject"][conversation.metadata.subject] += 1
            logger.event("validation_pass", example_id=path.stem, input=str(path))
        else:
            shutil.copy2(path, rejected_dir / path.name)
            rejected += 1
            issues = result.as_dict()["issues"] if result else [{"code": "exception", "message": "see errors.jsonl"}]
            logger.event("validation_fail", example_id=path.stem, input=str(path), issues=issues)

    summary = {
        "summary_stage": "validation",
        "accepted": accepted,
        "rejected": rejected,
        "run_llm_judge": run_llm_judge,
        "counts": {name: dict(counter) for name, counter in counters.items()},
    }
    write_summary(run_dir, summary)
    logger.event("validation_end", **summary)
    return 0 if rejected == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate generated tutoring SFT data.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--skip-llm-judge",
        action="store_true",
        help="Run only schema and deterministic local validation.",
    )
    args = parser.parse_args(argv)
    return validate_run(args.run_dir, args.config, run_llm_judge=not args.skip_llm_judge)


if __name__ == "__main__":
    raise SystemExit(main())

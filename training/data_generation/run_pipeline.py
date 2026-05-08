from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.convert import convert
from training.data_generation.export_reviewed import export_reviewed
from training.data_generation.generate_dataset import generate
from training.data_generation.logging_utils import RunLogger
from training.data_generation.schema import load_config
from training.data_generation.validate_dataset import validate_run


def run_pipeline(
    *,
    config_path: Path,
    export_output: Path | None,
    include_draft_valid: bool,
    convert_output: Path | None,
    convert_stats: Path | None,
    stop_on_validation_failure: bool,
) -> int:
    config = load_config(config_path)
    run_dir = config.run.output_dir
    logger = RunLogger(run_dir, "training.data_generation.pipeline")
    logger.event(
        "pipeline_start",
        config_path=str(config_path),
        run_dir=str(run_dir),
        export_output=str(export_output) if export_output else None,
        convert_output=str(convert_output) if convert_output else None,
    )
    if convert_output and not export_output:
        logger.event("pipeline_end", return_code=2, reason="convert_requires_export_output")
        return 2

    generation_rc = generate(config_path)
    logger.event("pipeline_generation_done", return_code=generation_rc)
    if generation_rc != 0:
        logger.event("pipeline_end", return_code=generation_rc, reason="generation_failed")
        return generation_rc

    validation_rc = validate_run(run_dir, run_llm_judge=False)
    logger.event("pipeline_validation_done", return_code=validation_rc)
    valid_count = len(list((run_dir / "valid").glob("*.json")))
    if valid_count < config.run.total_examples:
        logger.event(
            "final_target_shortfall",
            target=config.run.total_examples,
            valid=valid_count,
        )
        logger.event("pipeline_end", return_code=1, reason="final_target_shortfall")
        return 1
    if validation_rc != 0 and stop_on_validation_failure:
        logger.event("pipeline_end", return_code=validation_rc, reason="validation_failed")
        return validation_rc

    if export_output:
        export_rc = export_reviewed(
            run_dir=run_dir,
            output=export_output,
            include_draft_valid=include_draft_valid,
        )
        logger.event("pipeline_export_done", return_code=export_rc)
        if export_rc != 0:
            logger.event("pipeline_end", return_code=export_rc, reason="export_failed")
            return export_rc
        exported_count = len(list(export_output.glob("*.json")))
        if convert_output and exported_count < config.run.total_examples:
            logger.event(
                "export_target_shortfall",
                target=config.run.total_examples,
                exported=exported_count,
                include_draft_valid=include_draft_valid,
            )
            logger.event("pipeline_end", return_code=1, reason="export_target_shortfall")
            return 1
    else:
        export_rc = 0

    if convert_output:
        stats_path = convert_stats or convert_output.with_suffix(".stats.json")
        stats = convert(export_output, convert_output, stats_path)
        logger.event(
            "pipeline_convert_done",
            output=str(convert_output),
            stats=str(stats_path),
            total_conversations=stats["total_conversations"],
        )

    final_rc = validation_rc if validation_rc != 0 else export_rc
    logger.event("pipeline_end", return_code=final_rc, reason="complete")
    return final_rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate, validate, optionally export, and optionally convert synthetic SFT data."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--export-output",
        type=Path,
        default=None,
        help="Optional reviewed export directory. If omitted, pipeline stops after validation.",
    )
    parser.add_argument(
        "--include-draft-valid",
        action="store_true",
        help="Export valid draft examples too. Useful for smoke tests before manual approval.",
    )
    parser.add_argument(
        "--convert-output",
        type=Path,
        default=None,
        help="Optional SFT JSONL output path. Requires --export-output.",
    )
    parser.add_argument("--convert-stats", type=Path, default=None)
    parser.add_argument(
        "--keep-going-on-validation-failure",
        action="store_true",
        help="Continue to export valid examples even when some raw examples fail validation.",
    )
    args = parser.parse_args(argv)
    return run_pipeline(
        config_path=args.config,
        export_output=args.export_output,
        include_draft_valid=args.include_draft_valid,
        convert_output=args.convert_output,
        convert_stats=args.convert_stats,
        stop_on_validation_failure=not args.keep_going_on_validation_failure,
    )


if __name__ == "__main__":
    raise SystemExit(main())

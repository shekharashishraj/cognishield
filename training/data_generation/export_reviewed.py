"""Export validated generated examples for hand-review / SFT conversion.

Benchmark fields such as ``reference_solution`` under ``turn_context.task_context``
stay in exported JSON; ``training.convert`` only serializes chat ``messages`` into
JSONL unless you extend the converter.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.data_generation.logging_utils import RunLogger, write_summary
from training.data_generation.schema import GeneratedConversation, conversation_to_json


def export_reviewed(
    *,
    run_dir: Path,
    output: Path,
    include_draft_valid: bool = False,
    selected_file: Path | None = None,
) -> int:
    logger = RunLogger(run_dir, "training.data_generation.export")
    valid_dir = run_dir / "valid"
    output.mkdir(parents=True, exist_ok=True)
    for stale in output.glob("*.json"):
        stale.unlink()
    selected: set[str] | None = None
    if selected_file:
        selected = {
            line.strip()
            for line in selected_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
    exported = 0
    skipped = 0
    for path in sorted(valid_dir.glob("*.json")):
        conversation = GeneratedConversation.model_validate_json(path.read_text(encoding="utf-8"))
        should_export = conversation.annotator.review_status == "approved" or include_draft_valid
        if selected is not None:
            should_export = should_export or path.name in selected or path.stem in selected
        if not should_export:
            skipped += 1
            logger.event("export_skip", input=str(path), review_status=conversation.annotator.review_status)
            continue
        exported += 1
        filename = f"{exported:03d}.json"
        conversation.conversation_id = f"mt_{exported:03d}"
        out_path = output / filename
        out_path.write_text(conversation_to_json(conversation), encoding="utf-8")
        logger.event("export_write", input=str(path), output=str(out_path), conversation_id=conversation.conversation_id)
    summary = {"summary_stage": "export", "exported": exported, "skipped": skipped, "output": str(output)}
    write_summary(run_dir, summary)
    logger.event("export_end", **summary)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export reviewed generated examples for SFT conversion.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-draft-valid", action="store_true")
    parser.add_argument("--selected-file", type=Path, default=None)
    args = parser.parse_args(argv)
    return export_reviewed(
        run_dir=args.run_dir,
        output=args.output,
        include_draft_valid=args.include_draft_valid,
        selected_file=args.selected_file,
    )


if __name__ == "__main__":
    raise SystemExit(main())

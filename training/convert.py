"""Convert annotated multi-turn JSON files into an SFT JSONL dataset.

Reads every conversation under ``data/multi_turn/*.json``, prepends the
referenced system prompt from ``cognishield/app/prompts/`` and writes one
OpenAI-style ``{"messages": [...]}`` record per conversation to
``training/data/sft.jsonl``. Also writes ``training/data/sft.stats.json``
with split-mix counts so we can sanity-check the corpus at scale.

Validation (fails the run if violated):
- ``conversation_id`` matches the filename stem (e.g. ``001.json`` ->
  ``mt_001``), per docs/annotation_guidelines.md section 8.
- Every assistant turn has ``loss_mask: true``. Per-turn ``loss_mask`` is
  honored by TRL's ``assistant_only_loss=True`` only as long as this
  invariant holds; if you ever introduce assistant turns with
  ``loss_mask: false`` (e.g. deliberately wrong drafts for revision-style
  data), this script will refuse to emit them so the trainer is not
  silently misled.

Usage:
    python training/convert.py
    python training/convert.py --input data/multi_turn --output training/data/sft.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "multi_turn"
DEFAULT_OUTPUT = REPO_ROOT / "training" / "data" / "sft.jsonl"
DEFAULT_STATS = REPO_ROOT / "training" / "data" / "sft.stats.json"
PROMPTS_DIR = REPO_ROOT / "cognishield" / "app" / "prompts"


@dataclass
class ConversionError(Exception):
    """Raised when a conversation fails validation."""

    path: Path
    reason: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.path}: {self.reason}"


def load_system_prompt(prompt_id: str) -> str:
    """Resolve a ``prompt_id`` like ``primary.txt@v1`` to file contents.

    The ``@vN`` suffix is part of the spec but currently only one version
    exists per prompt, so it is parsed and ignored. Future versions can be
    routed by mapping ``(name, version) -> path`` here.
    """
    name, _, version = prompt_id.partition("@")
    if version and version != "v1":
        raise ValueError(
            f"Unsupported prompt version {version!r} in prompt_id {prompt_id!r}"
        )
    path = PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"System prompt not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def _expected_conversation_id(file_path: Path) -> str:
    return f"mt_{file_path.stem}"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _validate_record(file_path: Path, raw: dict) -> None:
    expected_id = _expected_conversation_id(file_path)
    actual_id = raw.get("conversation_id")
    if actual_id != expected_id:
        raise ConversionError(
            file_path,
            f"conversation_id={actual_id!r} does not match filename stem "
            f"(expected {expected_id!r}); see annotation_guidelines.md section 8",
        )

    messages = raw.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ConversionError(file_path, "messages must be a non-empty list")

    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role not in {"user", "assistant", "system"}:
            raise ConversionError(
                file_path, f"messages[{idx}].role={role!r} is not user/assistant/system"
            )
        if not isinstance(msg.get("content"), str):
            raise ConversionError(
                file_path, f"messages[{idx}].content must be a string"
            )
        if role == "assistant" and msg.get("loss_mask") is not True:
            raise ConversionError(
                file_path,
                f"messages[{idx}] is an assistant turn with loss_mask != true. "
                "TRL's assistant_only_loss=True trains every assistant span; if "
                "you intend to author a non-trained assistant draft, drop it "
                "into a separate revision-style dataset and add a custom "
                "collator before re-enabling this file.",
            )
        if role == "user" and msg.get("loss_mask") is not False:
            raise ConversionError(
                file_path,
                f"messages[{idx}] is a user turn with loss_mask != false; user "
                "spans must never contribute to the loss",
            )


def _to_openai_messages(raw: dict, system_prompt: str) -> list[dict]:
    """Build the OpenAI-format ``messages`` list with system prompt prepended."""
    out: list[dict] = [{"role": "system", "content": system_prompt}]
    for msg in raw["messages"]:
        out.append({"role": msg["role"], "content": msg["content"]})
    return out


def iter_conversations(input_dir: Path) -> Iterable[Path]:
    yield from sorted(input_dir.glob("*.json"))


def convert(input_dir: Path, output_path: Path, stats_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_counter: Counter[str] = Counter()
    coercion_counter: Counter[str] = Counter()
    policy_counter: Counter[str] = Counter()
    total_assistant_turns = 0
    written = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        for path in iter_conversations(input_dir):
            with path.open("r", encoding="utf-8") as in_f:
                raw = json.load(in_f)

            _validate_record(path, raw)

            prompt_id = raw.get("system_prompt", {}).get("prompt_id", "primary.txt@v1")
            system_prompt = load_system_prompt(prompt_id)

            messages = _to_openai_messages(raw, system_prompt)
            record = {
                "conversation_id": raw["conversation_id"],
                "split": raw.get("split", "unknown"),
                "messages": messages,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            written += 1
            split_counter[raw.get("split", "unknown")] += 1
            coercion_counter[raw.get("coercion_level", "unknown")] += 1
            policy_counter[
                raw.get("turn_context", {})
                .get("rubric_constraints", {})
                .get("tutor_answer_policy", "unknown")
            ] += 1
            total_assistant_turns += sum(
                1 for m in raw["messages"] if m.get("role") == "assistant"
            )

    stats = {
        "total_conversations": written,
        "total_assistant_turns": total_assistant_turns,
        "split": dict(split_counter),
        "coercion_level": dict(coercion_counter),
        "tutor_answer_policy": dict(policy_counter),
        "input_dir": _display_path(input_dir),
        "output": _display_path(output_path),
    }
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def _format_stats(stats: dict) -> str:
    lines = [
        f"Wrote {stats['total_conversations']} conversations "
        f"({stats['total_assistant_turns']} assistant turns) to {stats['output']}",
        "",
        "Split mix:",
    ]
    for k, v in sorted(stats["split"].items()):
        lines.append(f"  {k:<24} {v}")
    lines.append("Coercion level:")
    for k, v in sorted(stats["coercion_level"].items()):
        lines.append(f"  {k:<24} {v}")
    lines.append("Tutor answer policy:")
    for k, v in sorted(stats["tutor_answer_policy"].items()):
        lines.append(f"  {k:<24} {v}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    args = parser.parse_args(argv)

    if not args.input.is_dir():
        print(f"error: input directory not found: {args.input}", file=sys.stderr)
        return 2

    try:
        stats = convert(args.input, args.output, args.stats)
    except ConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(_format_stats(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

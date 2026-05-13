"""Phase 3 (optional): Rejection-Sampling Refresh (paper §5.3).

Replay GRPO rollouts (from `rollouts.jsonl` produced by training/rl/grpo_trainer.py),
keep top-quartile by total reward, and run one SFT epoch on the filtered set.
Reuses the SFT trainer + masked-loss collator unchanged — we just emit a new
`top_quartile.jsonl` in cognibench format first.

Usage:
    python -m training.rft.train \
        --rollouts checkpoints/tutor_grpo/rollouts.jsonl \
        --out_data rollouts/top_quartile.jsonl \
        --sft_config training/configs/sft/qwen35_9b_full.yaml \
        --override data.path=rollouts/top_quartile.jsonl \
                   optim.lr=1e-5 train.epochs=1 \
                   out_dir=checkpoints/tutor_rft
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import List


def select_top_quartile(rollouts_path: Path) -> List[dict]:
    rows = [json.loads(line) for line in rollouts_path.open() if line.strip()]
    if not rows:
        raise ValueError(f"no rollouts found in {rollouts_path}")
    totals = [r["reward"]["total"] for r in rows]
    cutoff = statistics.quantiles(totals, n=4)[2]  # 75th percentile
    return [r for r, t in zip(rows, totals) if t >= cutoff]


_ROLE_BACK = {"user": "student", "assistant": "teacher"}


def rollout_to_cognibench(r: dict) -> dict:
    """Convert a logged rollout into a cognibench.jsonl-shaped record."""
    transcript = r["transcript"]
    turns = []
    for i, m in enumerate(transcript):
        role = _ROLE_BACK.get(m["role"], m["role"])
        turns.append({"role": role, "content": m["content"], "turn_number": i + 1})
    spec = r.get("spec", {})
    return {
        "split": "rft_top_quartile",
        "split_label": "RFT: Top Quartile",
        "subject": r.get("subject") or "unknown",
        "problem": r["problem"],
        "solution": r.get("ground_truth", ""),
        "expected_behavior": "scaffold and hint",
        "coercion_level": {0: "none", 1: "low", 2: "moderate", 3: "high"}.get(
            spec.get("coercion_level", 0), "none"
        ),
        "age_band": spec.get("age_band"),
        "student_persona": spec.get("persona"),
        "turns": turns,
        "metadata": {"reward": r["reward"]},
        "judge_accepted": True,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pareto Tutors RFT (rejection-sampling refresh).")
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--out_data", required=True)
    ap.add_argument("--sft_config", required=True)
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args(argv)

    keep = select_top_quartile(Path(args.rollouts))
    out_path = Path(args.out_data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in keep:
            f.write(json.dumps(rollout_to_cognibench(r)) + "\n")
    print(f"[rft] wrote {len(keep)} top-quartile rollouts to {out_path}")

    # Reuse the SFT entrypoint with the new data path.
    from training.sft.train import main as sft_main

    sft_argv = ["--config", args.sft_config]
    for o in args.override:
        sft_argv += ["--override", o]
    sft_argv += ["--override", f"data.path={out_path}"]
    return sft_main(sft_argv)


if __name__ == "__main__":
    sys.exit(main())

"""Thin wrapper over `trl.GRPOTrainer` for the Pareto-Tutors recipe.

Each prompt fed to GRPO is a `{"problem", "ground_truth", "spec_seed"}` JSON
blob. The reward callable:
  1. Reads the spec_seed and samples a Spec.
  2. Runs a tutor↔student rollout, with the *current policy* as the tutor.
  3. Scores the transcript with the full Pareto reward stack.
  4. Returns the scalar reward.

NOTE: This deviates slightly from how `GRPOTrainer` is "normally" used. The
trainer generates a tutor completion from `prompt` and passes the completion
to `reward_funcs`. We don't actually use that completion — we re-run our own
multi-turn rollout using the trainer's current model. The trainer's own
single-shot completion is computed only because TRL's loop requires it; it
provides the log-probs used for the GRPO advantage estimate (since each
rollout's first tutor turn is the same prompt continuation, the log-probs
of the first turn carry the policy gradient signal).

Multi-turn credit assignment is conversation-level (paper §8.2) — TRL's
group-mean normalization handles it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from training.configs._schema import GRPOConfig
from training.judges.base import ChatClient
from training.judges.vllm_client import MockClient, VLLMClient
from training.rl.env import build_sample_from_rollout, compute_reward
from training.rl.rollout import run_rollout
from training.rl.spec_sampler import SpecSampler, Spec
from training.rl.student_sim import MockStudent
from training.rewards.base import RewardBreakdown


def _make_judge_client(cfg: GRPOConfig) -> ChatClient:
    if cfg.judges.mock:
        return MockClient('{"accept": true, "anti_leakage": true, "helpfulness_tone": true}')
    return VLLMClient(
        endpoint=cfg.judges.endpoint,
        model=cfg.judges.model,
        api_key=cfg.judges.api_key,
        timeout=cfg.judges.timeout,
    )


def _make_student_client(cfg: GRPOConfig) -> ChatClient:
    if cfg.student.mock:
        return MockStudent()
    return VLLMClient(
        endpoint=cfg.student.endpoint,
        model=cfg.student.model,
        api_key=cfg.student.api_key,
        timeout=cfg.student.timeout,
    )


def make_reward_func(
    cfg: GRPOConfig,
    tutor_fn: Callable[[List[Dict[str, str]]], str],
    judge_client: Optional[ChatClient] = None,
    student_client: Optional[ChatClient] = None,
    log_jsonl: Optional[Path] = None,
):
    """Build a TRL-compatible reward callable.

    TRL passes `(prompts, completions, **kwargs)` to `reward_funcs`. We
    ignore `completions` (which contain only the first-turn continuation)
    and use `prompts` (the problem spec) to drive a full multi-turn rollout.
    """
    judge_client = judge_client or _make_judge_client(cfg)
    student_client = student_client or _make_student_client(cfg)
    sampler = SpecSampler(seed=cfg.seed)

    def _reward_fn(prompts: List[str], completions: List[str], **_: Any) -> List[float]:
        rewards: List[float] = []
        for prompt in prompts:
            spec_blob = json.loads(prompt)
            problem = spec_blob["problem"]
            ground_truth = spec_blob["ground_truth"]
            spec = sampler.sample()

            rollout = run_rollout(
                tutor_fn=tutor_fn,
                student_client=student_client,
                problem=problem,
                spec=spec,
                max_turns=cfg.rollout.max_dialog_turns,
                min_turns=cfg.rollout.min_dialog_turns,
                end_token=cfg.rollout.end_token,
                student_temperature=cfg.student.temperature,
            )
            sample = build_sample_from_rollout(
                problem=problem,
                ground_truth=ground_truth,
                transcript=rollout.transcript,
                spec=spec,
            )
            breakdown: RewardBreakdown = compute_reward(
                sample, cfg=cfg, judge_client=judge_client, student_client=student_client
            )
            rewards.append(breakdown.total)

            if log_jsonl is not None:
                log_jsonl.parent.mkdir(parents=True, exist_ok=True)
                with log_jsonl.open("a") as f:
                    f.write(
                        json.dumps(
                            {
                                "problem": problem,
                                "ground_truth": ground_truth,
                                "subject": spec_blob.get("subject"),
                                "spec": {
                                    "persona": spec.persona,
                                    "age_band": spec.age_band,
                                    "coercion_level": spec.coercion_level,
                                },
                                "turns_taken": rollout.turns_taken,
                                "ended_naturally": rollout.ended_naturally,
                                "reward": breakdown.as_dict(),
                                "transcript": sample.transcript,
                            }
                        )
                        + "\n"
                    )
        return rewards

    return _reward_fn

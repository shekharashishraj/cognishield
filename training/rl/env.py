"""End-to-end environment glue: rollout → RolloutSample → reward breakdown.

`compute_reward` is the function GRPO will call once per group sample. It is
pure-Python and idempotent (modulo judge non-determinism at T>0); the trainer
treats it as a black-box scalar reward producer.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from training.configs._schema import GRPOConfig
from training.judges.base import ChatClient
from training.rewards.aggregator import aggregate
from training.rewards.age import age_reward
from training.rewards.auxiliary import auxiliary_reward
from training.rewards.base import RewardBreakdown, RolloutSample
from training.rewards.pedagogy import pedagogy_reward
from training.rewards.safety import safety_reward
from training.rewards.solve import solve_reward
from training.rl.spec_sampler import Spec


def build_sample_from_rollout(
    problem: str,
    ground_truth: str,
    transcript: List[Dict[str, str]],
    spec: Spec,
    *,
    domain: str = "math",
) -> RolloutSample:
    """Filter system messages out of the transcript before reward computation."""
    cleaned: List[Dict[str, str]] = [m for m in transcript if m["role"] in ("user", "assistant")]
    return RolloutSample(
        problem=problem,
        ground_truth=ground_truth,
        transcript=cleaned,
        domain=domain,
        age_band=spec.age_band,
        coercion_level=spec.coercion_level,
        persona=spec.persona,
    )


def compute_reward(
    sample: RolloutSample,
    *,
    cfg: GRPOConfig,
    judge_client: ChatClient,
    student_client: ChatClient,
    solve_k: Optional[int] = None,
) -> RewardBreakdown:
    """Run the full reward stack on a single rollout sample."""
    k = solve_k if solve_k is not None else cfg.student.solve_samples_k

    r_sol = solve_reward(sample, student_client, k=k)
    r_ped = pedagogy_reward(sample, judge_client, samples=2)
    r_safety = safety_reward(sample, judge_client)
    r_age = age_reward(sample, judge_client)
    r_aux = auxiliary_reward(
        sample,
        end_token=cfg.rollout.end_token,
        aux_template=cfg.rewards.aux_template,
        aux_eoc=cfg.rewards.aux_eoc,
        aux_length=cfg.rewards.aux_length,
        length_budget_tokens_per_turn=cfg.rewards.length_budget_tokens_per_turn,
    )

    return aggregate(
        r_sol=r_sol,
        r_ped=r_ped,
        r_safety=r_safety,
        r_age=r_age,
        r_aux=r_aux,
        weights=cfg.rewards,
    )

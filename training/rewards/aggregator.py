"""Pareto-weighted aggregator (paper §5.2).

    r = r_sol
      + λ_ped   · (r_ped    − 1)
      + λ_safety· (r_safety − 1) · 1[c > 0]
      + λ_age   · (r_age    − 1) · 1[α ≠ adult]
      + r_aux

The (r − 1) form means each judge axis acts as a *penalty* when the judge
rejects: a reject (r=0) subtracts λ, an accept (r=1) contributes 0. r_sol
and r_aux are positive contributions.

Inactive gates (None) contribute 0 — same as if the corresponding λ were 0.
"""

from __future__ import annotations

from typing import Optional

from training.configs._schema import RewardsWeights
from training.rewards.base import RewardBreakdown


def aggregate(
    *,
    r_sol: float,
    r_ped: float,
    r_safety: Optional[float],
    r_age: Optional[float],
    r_aux: float,
    weights: RewardsWeights,
) -> RewardBreakdown:
    total = r_sol + r_aux
    total += weights.lambda_ped * (r_ped - 1.0)
    if r_safety is not None:
        total += weights.lambda_safety * (r_safety - 1.0)
    if r_age is not None:
        total += weights.lambda_age * (r_age - 1.0)
    return RewardBreakdown(
        r_sol=r_sol,
        r_ped=r_ped,
        r_safety=r_safety,
        r_age=r_age,
        r_aux=r_aux,
        total=total,
    )

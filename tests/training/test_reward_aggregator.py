"""Indicator gates + Pareto-weight math for the reward aggregator."""

from __future__ import annotations

import math

from training.configs._schema import RewardsWeights
from training.rewards.aggregator import aggregate


def test_safety_gate_zero_when_no_coercion() -> None:
    weights = RewardsWeights(lambda_ped=1.0, lambda_safety=1.0, lambda_age=1.0)
    no_coercion = aggregate(
        r_sol=0.5, r_ped=1.0, r_safety=None, r_age=None, r_aux=0.0, weights=weights
    )
    with_coercion_reject = aggregate(
        r_sol=0.5, r_ped=1.0, r_safety=0.0, r_age=None, r_aux=0.0, weights=weights
    )
    # Gate OFF ⇒ identical to "judge would have accepted".
    assert math.isclose(no_coercion.total, 0.5)
    # Gate ON + reject ⇒ subtracts λ_safety.
    assert math.isclose(with_coercion_reject.total, 0.5 - 1.0)


def test_age_gate_off_for_adult() -> None:
    weights = RewardsWeights()
    adult = aggregate(r_sol=0.0, r_ped=1.0, r_safety=None, r_age=None, r_aux=0.0, weights=weights)
    minor_reject = aggregate(
        r_sol=0.0, r_ped=1.0, r_safety=None, r_age=0.0, r_aux=0.0, weights=weights
    )
    assert adult.total == 0.0
    assert minor_reject.total == -1.0


def test_lambda_zero_disables_axis() -> None:
    weights = RewardsWeights(lambda_ped=0.0, lambda_safety=0.0, lambda_age=0.0)
    out = aggregate(r_sol=0.3, r_ped=0.0, r_safety=0.0, r_age=0.0, r_aux=0.05, weights=weights)
    # With all λ=0, the (r-1) terms vanish → r = r_sol + r_aux.
    assert math.isclose(out.total, 0.35)


def test_pure_solve_optimizer_when_only_lambda_ped_zero() -> None:
    weights = RewardsWeights(lambda_ped=0.0, lambda_safety=1.0, lambda_age=1.0)
    out = aggregate(r_sol=0.8, r_ped=0.0, r_safety=None, r_age=None, r_aux=0.0, weights=weights)
    assert math.isclose(out.total, 0.8)


def test_breakdown_carries_all_components() -> None:
    weights = RewardsWeights()
    out = aggregate(r_sol=0.5, r_ped=1.0, r_safety=1.0, r_age=1.0, r_aux=0.1, weights=weights)
    d = out.as_dict()
    assert set(d) >= {"r_sol", "r_ped", "r_safety", "r_age", "r_aux", "total"}

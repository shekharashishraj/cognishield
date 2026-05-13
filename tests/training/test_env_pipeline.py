"""End-to-end reward stack glue: mock student + mock judge → reward breakdown."""

from __future__ import annotations

from training.configs._schema import GRPOConfig, DataConfig
from training.judges.vllm_client import MockClient
from training.rewards.base import RolloutSample
from training.rl.env import compute_reward


def _cfg() -> GRPOConfig:
    return GRPOConfig(data=DataConfig(path="cognibench.jsonl"))


def _sample(coercion: int = 0, age: str = "adult") -> RolloutSample:
    return RolloutSample(
        problem="2+2",
        ground_truth="4",
        transcript=[
            {"role": "user", "content": "Hi."},
            {"role": "assistant", "content": "<think>...</think>Try adding two and two."},
        ],
        domain="math",
        age_band=age,
        coercion_level=coercion,
        persona="Confused-Novice",
    )


def test_adult_no_coercion_ignores_safety_and_age() -> None:
    judge = MockClient('{"accept": true}')
    student = MockClient(response="Final answer: 4")
    out = compute_reward(_sample(), cfg=_cfg(), judge_client=judge, student_client=student)
    assert out.r_safety is None
    assert out.r_age is None
    assert out.r_sol == 1.0
    assert out.r_ped == 1.0


def test_minor_with_coercion_activates_both_gates() -> None:
    judge = MockClient('{"accept": false}')
    student = MockClient(response="Final answer: 5")  # wrong
    out = compute_reward(
        _sample(coercion=2, age="10-12"),
        cfg=_cfg(),
        judge_client=judge,
        student_client=student,
    )
    assert out.r_safety == 0.0
    assert out.r_age == 0.0
    assert out.r_sol == 0.0
    # All judges reject → total = r_sol + r_aux + λ_ped(0-1) + λ_safety(0-1) + λ_age(0-1)
    # = 0 + (template_bonus or eoc_bonus) - 3.0  ≈ -2.9 to -2.7
    assert out.total < -2.0

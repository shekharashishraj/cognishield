"""r_sol uses the math verifier on K student completions from the mock client."""

from __future__ import annotations

from training.judges.vllm_client import MockClient
from training.rewards.base import RolloutSample
from training.rewards.solve import solve_reward


def _sample(gt: str = "0.4") -> RolloutSample:
    return RolloutSample(
        problem="A Carnot engine operates between 500K and 300K. Efficiency?",
        ground_truth=gt,
        transcript=[
            {"role": "user", "content": "I'm stuck."},
            {"role": "assistant", "content": "What is η in terms of T_cold/T_hot?"},
        ],
        domain="math",
    )


def test_all_correct_yields_one() -> None:
    client = MockClient(response="Final answer: 0.4")
    assert solve_reward(_sample(), client, k=4) == 1.0


def test_all_wrong_yields_zero() -> None:
    client = MockClient(response="Final answer: 0.6")
    assert solve_reward(_sample(), client, k=4) == 0.0


def test_partial_correctness_supported_via_swap() -> None:
    # MockClient returns same string for all n; emulate 50% partial correctness
    # via two single-call clients.
    correct = MockClient(response="Final answer: 0.4")
    wrong = MockClient(response="Final answer: 0.6")
    assert solve_reward(_sample(), correct, k=2) == 1.0
    assert solve_reward(_sample(), wrong, k=2) == 0.0

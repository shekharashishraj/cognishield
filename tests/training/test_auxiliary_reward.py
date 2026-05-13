"""Template / EOC / length bonuses on the auxiliary reward."""

from __future__ import annotations

from training.rewards.auxiliary import auxiliary_reward
from training.rewards.base import RolloutSample


def _make_sample(turns):
    return RolloutSample(
        problem="p", ground_truth="g", transcript=turns, domain="math"
    )


def test_template_bonus_only_when_well_formed_think() -> None:
    well = _make_sample(
        [{"role": "assistant", "content": "<think>step 1</think>What's next?"}]
    )
    malformed = _make_sample([{"role": "assistant", "content": "<think>step 1 What's next?"}])
    plain = _make_sample([{"role": "assistant", "content": "What's next?"}])
    assert auxiliary_reward(well, aux_template=0.1, aux_eoc=0.0, aux_length=0.0) == 0.1
    assert auxiliary_reward(malformed, aux_template=0.1, aux_eoc=0.0, aux_length=0.0) == 0.0
    assert auxiliary_reward(plain, aux_template=0.1, aux_eoc=0.0, aux_length=0.0) == 0.0


def test_eoc_bonus_when_used() -> None:
    used = _make_sample([{"role": "assistant", "content": "Good work. <end_of_conversation>"}])
    not_used = _make_sample([{"role": "assistant", "content": "Good work."}])
    assert auxiliary_reward(used, aux_template=0.0, aux_eoc=0.2, aux_length=0.0) == 0.2
    assert auxiliary_reward(not_used, aux_template=0.0, aux_eoc=0.2, aux_length=0.0) == 0.0


def test_length_penalty_when_over_budget() -> None:
    long_turn = " ".join(["word"] * 600)
    sample = _make_sample([{"role": "assistant", "content": long_turn}])
    out = auxiliary_reward(
        sample,
        aux_template=0.0,
        aux_eoc=0.0,
        aux_length=0.1,
        length_budget_tokens_per_turn=500,
    )
    assert out < 0  # over budget ⇒ small negative

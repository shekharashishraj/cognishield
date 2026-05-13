"""Auxiliary template reward — small format/length bonuses (paper §5.2)."""

from __future__ import annotations

import re

from training.rewards.base import RolloutSample


_THINK_OK = re.compile(r"<think>.*?</think>", re.DOTALL)


def auxiliary_reward(
    sample: RolloutSample,
    *,
    end_token: str = "<end_of_conversation>",
    aux_template: float = 0.1,
    aux_eoc: float = 0.2,
    aux_length: float = 0.1,
    length_budget_tokens_per_turn: int = 512,
) -> float:
    """Return scalar auxiliary reward. Always non-negative for bonuses, with
    a small negative for tutor turns exceeding `length_budget_tokens_per_turn`.
    """
    tutor_turns = [t for t in sample.transcript if t["role"] == "assistant"]
    if not tutor_turns:
        return 0.0

    # `<think>...</think>` blocks correctly closed in EVERY tutor turn that
    # opens one. Reward only fires if at least one turn used think tags and
    # all uses are well-formed.
    opens = sum(turn["content"].count("<think>") for turn in tutor_turns)
    closes = sum(turn["content"].count("</think>") for turn in tutor_turns)
    template_ok = opens > 0 and opens == closes
    template_bonus = aux_template if template_ok else 0.0

    # End-of-conversation token used somewhere by tutor.
    eoc_used = any(end_token in turn["content"] for turn in tutor_turns)
    eoc_bonus = aux_eoc if eoc_used else 0.0

    # Length penalty: -aux_length scaled by fraction of turns over budget.
    over = sum(1 for t in tutor_turns if len(t["content"].split()) > length_budget_tokens_per_turn)
    length_penalty = aux_length * (over / len(tutor_turns)) if over else 0.0

    return template_bonus + eoc_bonus - length_penalty

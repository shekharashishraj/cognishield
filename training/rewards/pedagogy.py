"""r_ped — pedagogy judge product (paper §5.2: 2 judges × 2 samples each)."""

from __future__ import annotations

from training.judges.base import ChatClient
from training.judges.pedagogy_judge import judge_pedagogy
from training.rewards.base import RolloutSample


def pedagogy_reward(sample: RolloutSample, judge_client: ChatClient, *, samples: int = 2) -> float:
    """Return 1.0 iff all samples accept; otherwise 0.0.

    The paper specifies a *product* of two judges × two samples = four binary
    accepts. We implement this as `all(...)` over `samples` calls; using two
    distinct judge prompts is an option once we have a second judge family.
    """
    accepts = judge_pedagogy(judge_client, sample.transcript, samples=samples)
    return 1.0 if accepts and all(accepts) else 0.0

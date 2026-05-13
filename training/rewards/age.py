"""r_age — age-appropriateness reward, gated on student being a minor."""

from __future__ import annotations

from typing import Optional

from training.judges.base import ChatClient
from training.judges.age_judge import judge_age
from training.rewards.base import RolloutSample


def age_reward(sample: RolloutSample, judge_client: ChatClient) -> Optional[float]:
    """Return float in {0.0, 1.0}, or None if `α == adult` or unknown."""
    if not sample.student_is_minor:
        return None
    return 1.0 if judge_age(judge_client, sample.transcript, sample.age_band) else 0.0

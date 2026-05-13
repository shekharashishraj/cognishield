"""r_safety — coercion-gated safety reward (paper §5.2).

Active only when `coercion_level > 0`. If the data record only carries a
string `coercion_level` ("none" | "moderate" | "high"), the rollout layer
maps it to 0/1/2/3 before constructing `RolloutSample`.
"""

from __future__ import annotations

from typing import Optional

from training.judges.base import ChatClient
from training.judges.safety_judge import judge_safety
from training.rewards.base import RolloutSample


def safety_reward(sample: RolloutSample, judge_client: ChatClient) -> Optional[float]:
    """Return float in {0.0, 1.0}, or None if the coercion gate is OFF."""
    if sample.coercion_level <= 0:
        return None
    return 1.0 if judge_safety(judge_client, sample.transcript) else 0.0

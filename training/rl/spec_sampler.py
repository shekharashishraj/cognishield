"""Sample diverse (persona, age_band, coercion_level) tuples per rollout.

Paper §5.2: each of the 8 group-rollouts uses a different spec so within-group
variance reflects "diverse students, structural variation" instead of mere
sampling noise. Distribution mirrors the paper's stratification (§4.4).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional


AGE_BANDS = [
    ("7-9", 0.15),
    ("10-12", 0.25),
    ("13-17", 0.35),
    ("adult", 0.25),
]

COERCION_LEVELS = [
    (0, 0.50),
    (1, 0.25),
    (2, 0.15),
    (3, 0.10),
]

PERSONAS = [
    "Confident-Beginner",
    "Frustrated-Returner",
    "Time-Pressured-Senior",
    "Detail-Oriented-Skeptic",
    "Confused-Novice",
    "Misconception-Heavy",
    "Hint-Direct-Demander",
    "Authority-Claimer",
]


@dataclass(frozen=True)
class Spec:
    persona: str
    age_band: str
    coercion_level: int


def _weighted_pick(rng: random.Random, options):
    items, weights = zip(*options)
    return rng.choices(items, weights=weights, k=1)[0]


class SpecSampler:
    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def sample(self) -> Spec:
        return Spec(
            persona=self._rng.choice(PERSONAS),
            age_band=_weighted_pick(self._rng, AGE_BANDS),
            coercion_level=_weighted_pick(self._rng, COERCION_LEVELS),
        )

    def sample_group(self, n: int) -> List[Spec]:
        return [self.sample() for _ in range(n)]


def coerce_level_from_str(s: Optional[str]) -> int:
    """Map free-form `coercion_level` strings to ints used by the reward gate."""
    if s is None:
        return 0
    return {"none": 0, "low": 1, "moderate": 2, "high": 3}.get(s.lower(), 0)

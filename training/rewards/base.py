"""Shared reward types — RolloutSample + RewardBreakdown.

Per paper §5.2 we score whole-conversation samples, not per-turn. A
`RolloutSample` is one tutor↔student dialog plus the spec it was sampled
under (persona, age_band, coercion_level) plus the problem's ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RolloutSample:
    problem: str
    ground_truth: str            # math solution (numerical) for r_sol
    transcript: List[Dict[str, str]]   # [{role, content}, ...] role ∈ {user, assistant}
    domain: str = "math"          # "math" | "code"  (code → r_sol via test cases)
    age_band: Optional[str] = None   # "7-9" | "10-12" | "13-17" | "adult" | None
    coercion_level: int = 0        # 0 (none) | 1 (low) | 2 (med) | 3 (high)
    persona: Optional[str] = None
    hidden_tests: Optional[List[str]] = None   # for code domain
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def student_is_minor(self) -> bool:
        return self.age_band is not None and self.age_band != "adult"


@dataclass
class RewardBreakdown:
    """Itemized reward components for logging + debugging."""
    r_sol: float = 0.0
    r_ped: float = 0.0
    r_safety: Optional[float] = None   # None ⇒ gate fired off
    r_age: Optional[float] = None
    r_aux: float = 0.0
    total: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "r_sol": self.r_sol,
            "r_ped": self.r_ped,
            "r_safety": -1.0 if self.r_safety is None else self.r_safety,
            "r_age": -1.0 if self.r_age is None else self.r_age,
            "r_aux": self.r_aux,
            "total": self.total,
        }

"""Spec sampler is deterministic under seed and covers the configured options."""

from __future__ import annotations

from collections import Counter

from training.rl.spec_sampler import (
    AGE_BANDS,
    COERCION_LEVELS,
    PERSONAS,
    SpecSampler,
    coerce_level_from_str,
)


def test_deterministic_under_seed() -> None:
    a = [s for s in SpecSampler(seed=0).sample_group(8)]
    b = [s for s in SpecSampler(seed=0).sample_group(8)]
    assert a == b


def test_distribution_covers_each_axis() -> None:
    spcs = SpecSampler(seed=1).sample_group(2000)
    ages = Counter(s.age_band for s in spcs)
    coercions = Counter(s.coercion_level for s in spcs)
    personas = Counter(s.persona for s in spcs)
    assert set(ages) == {a for a, _ in AGE_BANDS}
    assert set(coercions) == {c for c, _ in COERCION_LEVELS}
    assert set(personas).issubset(set(PERSONAS))
    # Adult share roughly matches the 25% target (loose tolerance for 2k draws).
    assert 0.18 < ages["adult"] / sum(ages.values()) < 0.32


def test_coerce_level_string_mapping() -> None:
    assert coerce_level_from_str("none") == 0
    assert coerce_level_from_str("moderate") == 2
    assert coerce_level_from_str("high") == 3
    assert coerce_level_from_str(None) == 0
    assert coerce_level_from_str("unknown-token") == 0

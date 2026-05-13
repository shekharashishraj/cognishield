"""Verify numerical exact-match handles the awkward formats real models emit."""

from __future__ import annotations

from training.verifiers.math import math_correct


def test_boxed_format() -> None:
    assert math_correct(r"The answer is \boxed{0.4}.", "0.4")
    assert not math_correct(r"\boxed{0.5}", "0.4")


def test_final_answer_prefix() -> None:
    # GT comes back as a clean number once the data pipeline normalizes;
    # tolerance is rtol=1e-3 so 7.40278 vs 7.4022 falls inside.
    assert math_correct("Working... Final answer: 7.40278", "7.4022")
    assert math_correct("Answer: 120/1024", "0.117")
    assert not math_correct("Final answer: 9", "10")


def test_fraction_and_pi_notation() -> None:
    assert math_correct("3/4", "0.75")
    assert math_correct("0.4", "2/5")
    # symbolic eval handles 'pi'
    assert math_correct("Final answer: pi", "3.14159265")


def test_substring_fallback() -> None:
    # Non-numeric exact match falls back to case-insensitive string compare.
    assert math_correct("Answer: linearly dependent", "linearly dependent")
    assert not math_correct("Answer: linearly independent", "linearly dependent")

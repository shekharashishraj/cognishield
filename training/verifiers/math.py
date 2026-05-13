"""Numerical exact-match verifier for math `r_sol` (BigMath-style)."""

from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Optional


_NUMBER_RE = re.compile(
    r"-?\d+\.\d+|-?\d+/\d+|-?\d+|-?\.\d+"
)

_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_FINAL_RE = re.compile(r"(?:final\s+answer|answer)\s*[:=]\s*([^\n]+)", re.IGNORECASE)


def _strip_units(s: str) -> str:
    return re.sub(r"[^\d./\-+%πpi*eE]", "", s).strip()


def _extract_candidate(text: str) -> Optional[str]:
    """Pull the most likely final-answer substring from a free-form response."""
    m = _BOXED_RE.findall(text)
    if m:
        return m[-1].strip()
    m = _FINAL_RE.findall(text)
    if m:
        return m[-1].strip()
    nums = _NUMBER_RE.findall(text)
    if nums:
        return nums[-1]
    return None


def _to_float(s: str) -> Optional[float]:
    s = s.strip()
    if not s:
        return None
    try:
        if "/" in s and all(part.lstrip("-").isdigit() for part in s.split("/")):
            return float(Fraction(s))
    except (ValueError, ZeroDivisionError):
        pass
    try:
        return float(s)
    except ValueError:
        pass
    # Symbolic eval. Use a regex pass to coerce `π` and bare `pi` into the
    # named constant `_pi`, avoiding clobbering when the source already
    # has explicit multiplication (`*pi`).
    expr = s.replace("π", "_pi")
    expr = re.sub(r"\bpi\b", "_pi", expr, flags=re.IGNORECASE)
    expr = expr.replace("^", "**")
    try:
        safe_globals = {
            "__builtins__": {},
            "_pi": math.pi,
            "e": math.e,
            "sqrt": math.sqrt,
        }
        return float(eval(expr, safe_globals, {}))  # noqa: S307 - sandboxed
    except Exception:
        return None


def math_correct(prediction: str, ground_truth: str, rtol: float = 1e-2, atol: float = 1e-3) -> bool:
    """True iff `prediction` numerically matches `ground_truth`.

    Robust to: \\boxed{...}, "final answer: X", trailing units, fractions,
    pi notation. Falls back to substring equality if numeric parsing fails.
    """
    cand = _extract_candidate(prediction) or prediction
    gt = _extract_candidate(ground_truth) or ground_truth
    pf = _to_float(_strip_units(cand))
    gf = _to_float(_strip_units(gt))
    if pf is not None and gf is not None:
        if math.isclose(pf, gf, rel_tol=rtol, abs_tol=atol):
            return True
    return cand.strip().lower() == gt.strip().lower()

"""Sandboxed unit-test verifier for code `r_sol`. v2 — math-only for v1."""

from __future__ import annotations

from typing import List


def code_pass_rate(prediction_code: str, hidden_tests: List[str]) -> float:
    """Run `hidden_tests` against `prediction_code`, return fraction passing.

    NOT IMPLEMENTED in v1 (math-only scope). The full v2 implementation
    will run each test inside a `subprocess` with `resource.setrlimit` for
    CPU/memory caps, no network, and a read-only tmpdir.
    """
    raise NotImplementedError("Code verifier deferred to v2.")  # TODO(v2)

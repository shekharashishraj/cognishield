from __future__ import annotations

from cognishield.app.orchestrator import run_turn
from cognishield.app.settings import Settings
from cognishield.app.state import init_state


def test_dry_run_returns_candidate_without_network() -> None:
    settings = Settings(dry_run=True)
    state = init_state(user_query="Solve my homework exactly.")
    out = run_turn(state, settings, tracer=None)
    assert "[dry_run]" in out
    assert state.plan is not None
    assert state.validator_reports

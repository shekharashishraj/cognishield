from __future__ import annotations

from cognishield.app.orchestrator import run_turn
from cognishield.app.settings import Settings
from cognishield.app.state import init_state


def test_dry_run_returns_candidate_without_network() -> None:
    settings = Settings(dry_run=True, pipeline="legacy")
    state = init_state(user_query="Solve my homework exactly.")
    out = run_turn(state, settings, tracer=None)
    assert "[dry_run]" in out
    assert state.plan is not None
    assert state.validator_reports


def test_dry_run_meta_pipeline_without_network() -> None:
    settings = Settings(dry_run=True, pipeline="meta")
    state = init_state(user_query="Help me with this problem.")
    out = run_turn(state, settings, tracer=None)
    assert "[dry_run]" in out
    assert state.primary_draft is not None
    assert state.meta_output is not None
    assert state.meta_verifier_decision is not None
    assert state.final_response_text == out

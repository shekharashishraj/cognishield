"""Rollout driver: alternating tutor/student turns, ends on end_token or max_turns."""

from __future__ import annotations

from training.rl.rollout import run_rollout
from training.rl.spec_sampler import Spec
from training.rl.student_sim import MockStudent


def test_rollout_ends_on_end_token_after_min_turns() -> None:
    # Tutor that immediately tries to end early — should be IGNORED until min_turns.
    tutor_replies = iter(
        [
            "First reply (no end).",
            "Second reply <end_of_conversation>",   # would end at turn 2, but min=4
            "Third reply (no end).",
            "Fourth reply <end_of_conversation>",   # ends at turn 4
        ]
    )

    def tutor_fn(transcript):
        return next(tutor_replies)

    student = MockStudent()
    result = run_rollout(
        tutor_fn=tutor_fn,
        student_client=student,
        problem="What is 2+2?",
        spec=Spec(persona="Confused-Novice", age_band="adult", coercion_level=0),
        max_turns=12,
        min_turns=4,
    )
    # Tutor turn counted at each replied step. min_turns=4 means tutor must
    # speak at least 4 times *and* the 4th (or later) reply contains the end
    # token. Each iter does 2 turns (tutor + student) unless end_token fires.
    assert result.ended_naturally
    # 4 tutor turns + 3 student turns (the last student turn doesn't fire).
    assert result.turns_taken >= 4


def test_rollout_caps_at_max_turns_when_no_end_token() -> None:
    def tutor_fn(transcript):
        return "Keep going..."

    student = MockStudent()
    result = run_rollout(
        tutor_fn=tutor_fn,
        student_client=student,
        problem="What is 2+2?",
        spec=Spec(persona="Confused-Novice", age_band="adult", coercion_level=0),
        max_turns=6,
        min_turns=2,
    )
    assert not result.ended_naturally
    assert result.turns_taken == 6


def test_transcript_includes_system_then_alternates() -> None:
    def tutor_fn(transcript):
        return "Tutor turn."

    student = MockStudent(replies=["Student turn."])
    result = run_rollout(
        tutor_fn=tutor_fn,
        student_client=student,
        problem="x",
        spec=Spec(persona="P", age_band="adult", coercion_level=0),
        max_turns=2,
        min_turns=1,
    )
    # Layout: [system, system_meta, user (opening), assistant, user, assistant]
    roles = [m["role"] for m in result.transcript]
    assert roles[0] == "system"
    assert roles[1] == "system"
    assert roles[2] == "user"
    # After the initial opening, tutor/student alternation.
    assert roles[3] == "assistant"

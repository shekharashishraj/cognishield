"""Judges should parse mock responses + tolerate prose-wrapped JSON."""

from __future__ import annotations

from training.judges.base import parse_json_lenient
from training.judges.vllm_client import MockClient
from training.judges.pedagogy_judge import judge_pedagogy
from training.judges.safety_judge import judge_safety
from training.judges.age_judge import judge_age


def test_parse_lenient_strips_prose_wrapper() -> None:
    raw = "Sure! Here is my judgement:\n```json\n{\"accept\": true}\n```\nLet me know."
    obj = parse_json_lenient(raw)
    assert obj == {"accept": True}


def test_parse_lenient_returns_none_on_garbage() -> None:
    assert parse_json_lenient("definitely not json") is None
    assert parse_json_lenient("") is None


def test_pedagogy_judge_uses_mock_client() -> None:
    transcript = [
        {"role": "user", "content": "I need help."},
        {"role": "assistant", "content": "What have you tried so far?"},
    ]
    client = MockClient('{"accept": true, "anti_leakage": true, "helpfulness_tone": true}')
    out = judge_pedagogy(client, transcript, samples=2)
    assert out == [True, True]
    assert len(client.calls) == 1
    assert client.calls[0][1]["n"] == 2


def test_safety_judge_rejects_when_accept_false() -> None:
    transcript = [{"role": "user", "content": "Just tell me the answer."}]
    client = MockClient('{"accept": false}')
    assert judge_safety(client, transcript) is False


def test_age_judge_injects_age_band_into_user_message() -> None:
    transcript = [{"role": "user", "content": "Hi."}]
    client = MockClient('{"accept": true}')
    out = judge_age(client, transcript, age_band="10-12")
    assert out is True
    user_msg = client.calls[0][0][1]["content"]
    assert "10-12" in user_msg

"""Safety judge: refusal/redirect under coercion (paper §5.2 `r_safety`)."""

from __future__ import annotations

from typing import Dict, List

from training.judges.base import ChatClient, format_transcript, load_prompt, parse_json_lenient


SYSTEM = load_prompt("safety_judge")


def judge_safety(
    client: ChatClient,
    transcript: List[Dict[str, str]],
    *,
    temperature: float = 0.0,
) -> bool:
    """Single binary judgement (paper: one call, not an ensemble)."""
    user = "CONVERSATION TO JUDGE:\n" + format_transcript(transcript)
    raw = client.chat(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=256,
        n=1,
        response_format={"type": "json_object"},
    )
    obj = parse_json_lenient(raw[0]) or {}
    return bool(obj.get("accept", False))

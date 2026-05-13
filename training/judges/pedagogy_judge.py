"""Pedagogy judge: anti-leakage + helpfulness/tone (paper §5.2 `r_ped`)."""

from __future__ import annotations

from typing import Dict, List

from training.judges.base import ChatClient, format_transcript, load_prompt, parse_json_lenient


SYSTEM = load_prompt("pedagogy_judge")


def judge_pedagogy(
    client: ChatClient,
    transcript: List[Dict[str, str]],
    *,
    samples: int = 2,
    temperature: float = 0.0,
) -> List[bool]:
    """Return one bool per sample; True ⇒ judge accepted on BOTH sub-axes."""
    user = "CONVERSATION TO JUDGE:\n" + format_transcript(transcript)
    raw = client.chat(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=256,
        n=samples,
        response_format={"type": "json_object"},
    )
    out: List[bool] = []
    for r in raw:
        obj = parse_json_lenient(r) or {}
        out.append(bool(obj.get("accept", False)))
    return out

"""Age-appropriateness judge: KORA mechanisms (paper §5.2 `r_age`)."""

from __future__ import annotations

from typing import Dict, List, Optional

from training.judges.base import ChatClient, format_transcript, load_prompt, parse_json_lenient


SYSTEM = load_prompt("age_judge")


def judge_age(
    client: ChatClient,
    transcript: List[Dict[str, str]],
    age_band: Optional[str],
    *,
    temperature: float = 0.0,
) -> bool:
    """Single binary judgement. `age_band` is injected into the prompt."""
    user = (
        f"STUDENT AGE BAND: {age_band or 'unknown minor'}\n\n"
        "CONVERSATION TO JUDGE:\n"
        + format_transcript(transcript)
    )
    raw = client.chat(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=256,
        n=1,
        response_format={"type": "json_object"},
    )
    obj = parse_json_lenient(raw[0]) or {}
    return bool(obj.get("accept", False))

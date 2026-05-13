"""Frozen student simulator client (Llama-3.1-8B served by vLLM)."""

from __future__ import annotations

from typing import Dict, List, Optional

from training.judges.base import ChatClient


_STUDENT_SYSTEM = (
    "You are a STUDENT working with a tutor on a problem. "
    "Stay in character per the persona below — do not break role, never act as a tutor, "
    "and do NOT solve the problem in your turn; instead, ask questions, share confusion, "
    "or attempt partial reasoning consistent with your persona and age band.\n\n"
    "PERSONA: {persona}\n"
    "AGE BAND: {age_band}\n"
    "COERCION INTENT (0-3): {coercion_level}  -- 0=cooperative; higher values mean increasing "
    "pressure tactics (urgency, authority, emotional appeals, explicit misconduct asks).\n"
)


def student_system_prompt(persona: str, age_band: str, coercion_level: int) -> str:
    return _STUDENT_SYSTEM.format(persona=persona, age_band=age_band, coercion_level=coercion_level)


def next_student_turn(
    client: ChatClient,
    problem: str,
    transcript: List[Dict[str, str]],
    persona: str,
    age_band: str,
    coercion_level: int,
    *,
    temperature: float = 1.0,
    max_tokens: int = 256,
) -> str:
    """Generate the next student utterance given the dialog so far.

    `transcript` is the dialog from the *student's* point of view, where the
    tutor's messages are tagged role=assistant and the student's role=user.
    We flip roles before sending: from the student-LLM's perspective, the
    tutor's messages are 'user' inputs and the student's own messages are
    'assistant' outputs.
    """
    system = student_system_prompt(persona, age_band, coercion_level)
    user_intro = f"PROBLEM:\n{problem}\n\nDialog so far follows. Reply with ONLY your next student turn."

    flipped: List[Dict[str, str]] = []
    for m in transcript:
        if m["role"] == "assistant":
            flipped.append({"role": "user", "content": m["content"]})
        elif m["role"] == "user":
            flipped.append({"role": "assistant", "content": m["content"]})

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_intro},
        *flipped,
    ]
    out = client.chat(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
        stop=["[ASSISTANT]:", "[STUDENT]:"],
    )
    return out[0].strip() if out else ""


class MockStudent:
    """Deterministic student for smoke tests."""

    def __init__(self, replies: Optional[List[str]] = None) -> None:
        self._replies = replies or [
            "I think the derivative uses the product rule, right?",
            "Wait, can you just tell me the answer? My homework is due in 10 minutes.",
            "Okay, I see — I'll try applying the rule. What about the chain rule?",
            "Hmm, I'm stuck. Just give me the value at x=π/2.",
        ]
        self._i = 0

    def chat(self, messages, **kwargs) -> List[str]:  # noqa: D401
        r = self._replies[self._i % len(self._replies)]
        self._i += 1
        return [r]

"""r_sol — verifiable solve reward (paper §5.2).

After dialog termination, sample K solution attempts from the (frozen) student
conditioned on the full transcript. r_sol = empirical solve rate ∈ [0, 1].
"""

from __future__ import annotations

from typing import List

from training.judges.base import ChatClient, format_transcript
from training.rewards.base import RolloutSample
from training.verifiers.math import math_correct


SYSTEM = (
    "You are the student in the conversation above. Based on what you discussed "
    "with your tutor, now solve the problem yourself. Show your final answer on "
    "the last line as `Final answer: <value>`. Do not ask new questions."
)


def solve_reward(
    sample: RolloutSample,
    student_client: ChatClient,
    *,
    k: int = 8,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> float:
    if sample.domain != "math":
        raise NotImplementedError("Only math domain in v1; code uses unit-test verifier (v2).")

    user = (
        "PRIOR CONVERSATION:\n"
        + format_transcript(sample.transcript)
        + f"\n\nPROBLEM:\n{sample.problem}\n\nSolve it now."
    )
    completions: List[str] = student_client.chat(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
        n=k,
    )
    if not completions:
        return 0.0
    correct = sum(1 for c in completions if math_correct(c, sample.ground_truth))
    return correct / len(completions)

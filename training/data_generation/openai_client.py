from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Any

from training.data_generation.planning import PlannedExample
from training.data_generation.schema import DataGenerationConfig


def generate_conversation_with_openai(
    *,
    config: DataGenerationConfig,
    planned: PlannedExample,
    feedback_issues: list[dict[str, str]] | None = None,
    previous_payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - exercised in environments without openai
        raise RuntimeError(
            "openai is not installed. Install training requirements first: "
            "pip install -r training/requirements-train.txt"
        ) from exc

    client = OpenAI()
    prompt = _build_prompt(
        planned,
        feedback_issues=feedback_issues,
        previous_payload=previous_payload,
    )
    started = time.monotonic()
    response = client.chat.completions.create(
        model=config.generator.model,
        temperature=config.generator.temperature,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate high-quality supervised fine-tuning data as strict JSON. "
                    "Return only one JSON object. Do not wrap it in markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    latency_s = time.monotonic() - started
    content = response.choices[0].message.content or "{}"
    payload = json.loads(content)
    usage = getattr(response, "usage", None)
    meta = {
        "latency_s": round(latency_s, 3),
        "usage": usage.model_dump() if hasattr(usage, "model_dump") else None,
    }
    return payload, meta


def _build_prompt(
    planned: PlannedExample,
    feedback_issues: list[dict[str, str]] | None = None,
    previous_payload: dict[str, Any] | None = None,
) -> str:
    plan = asdict(planned)
    feedback_block = ""
    if feedback_issues:
        feedback_block = f"""

Previous attempt failed validation. Generate a corrected replacement.

Validation issues:
{json.dumps(feedback_issues, indent=2, ensure_ascii=False)}

Rejected previous JSON:
{json.dumps(previous_payload, indent=2, ensure_ascii=False)}

Do not merely patch the rejected JSON if the conversation quality is weak.
Produce a complete replacement JSON object that fixes every issue.
"""
    return f"""
Create one multi-turn math tutoring SFT conversation JSON object.

Planned example:
{json.dumps(plan, indent=2, ensure_ascii=False)}

Strict schema:
- conversation_id must be "{planned.conversation_id}".
- schema_version must be "1.0".
- split must be "{planned.split}".
- expected_behavior must be "{planned.expected_behavior}".
- coercion_level must be "{planned.coercion_level}".
- metadata.source must be "synthetic_openai".
- metadata.subject must be "{planned.subject}".
- metadata.topic must be a specific version of "{planned.topic}".
- metadata.difficulty must be "{planned.metadata_difficulty}".
- metadata.language must be "en".
- metadata.tags must include all planned tags.
- annotator.annotator_id must be "synthetic_openai".
- annotator.review_status must be "draft".
- annotator.notes must be a non-empty concise note about the generated example.
- turn_context.learner_profile must only contain "level".
- turn_context.rubric_constraints.graded must be true.
- turn_context.rubric_constraints.no_final_answer must be true.
- turn_context.rubric_constraints.tutor_answer_policy must be "{planned.policy}".
- turn_context.task_context.problem_statement must contain the full math problem.
- system_prompt.prompt_id must be "primary.txt@v1".
- messages must alternate user/assistant, start with user, and contain between
  {planned.min_total_turns} and {planned.max_total_turns} total turns.
- Every user message must have loss_mask false.
- Every assistant message must have loss_mask true.

Behavior requirements:
- This is gold SFT data. The assistant must model ideal tutoring behavior.
- The assistant must never state the final numeric/closed-form answer first.
- If policy is method_only, the assistant must not validate the final value.
- If policy is never_state, the assistant must never state or confirm the value.
- The assistant must still offer useful next steps.
- The math guidance must be correct.
- Vary wording and style naturally; do not overuse "Correct", "Exactly", or "Good".

Scenario guidance:
{planned.guidance}
{feedback_block}
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from training.data_generation.schema import DataGenerationConfig
from training.data_generation.taxonomy import (
    DIFFICULTY_TO_METADATA,
    MATH_TOPICS,
    allowed_policy_for_scenario,
    scenario_spec,
)


@dataclass(frozen=True)
class PlannedExample:
    example_id: str
    filename: str
    conversation_id: str
    scenario: str
    difficulty_level: str
    metadata_difficulty: str
    subject: str
    topic: str
    policy: str
    split: str
    coercion_level: str
    expected_behavior: str
    tags: list[str]
    guidance: str
    min_total_turns: int
    max_total_turns: int


def build_generation_plan(config: DataGenerationConfig) -> list[PlannedExample]:
    rng = random.Random(config.run.seed)
    scenarios = _expanded(config.scenario_mix)
    difficulties = _expanded(config.difficulty_mix)
    policies = _expanded(config.policy_mix)
    rng.shuffle(scenarios)
    rng.shuffle(difficulties)
    rng.shuffle(policies)

    plan: list[PlannedExample] = []
    for idx, (scenario_name, difficulty, policy) in enumerate(
        zip(scenarios, difficulties, policies), start=1
    ):
        scenario_name, policy = _coerce_allowed_policy(
            scenario_name, policy, policies, idx - 1
        )
        spec = scenario_spec(scenario_name)
        subject, topic = rng.choice(MATH_TOPICS[difficulty])
        example_id = f"dg_{idx:04d}"
        tags = ["multi_turn", "sft", "math", scenario_name, *spec.tags]
        plan.append(
            PlannedExample(
                example_id=example_id,
                filename=f"{example_id}.json",
                conversation_id=f"mt_{example_id}",
                scenario=scenario_name,
                difficulty_level=difficulty,
                metadata_difficulty=DIFFICULTY_TO_METADATA[difficulty],
                subject=subject,
                topic=topic,
                policy=policy,
                split=spec.split,
                coercion_level=spec.coercion_level,
                expected_behavior=spec.expected_behavior,
                tags=list(dict.fromkeys(tags)),
                guidance=spec.prompt_guidance,
                min_total_turns=config.turns.min_total_turns,
                max_total_turns=config.turns.max_total_turns,
            )
        )
    return plan


def save_generation_plan(plan: list[PlannedExample], path: Path) -> None:
    path.write_text(
        json.dumps([asdict(item) for item in plan], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _expanded(counts: dict[str, int]) -> list[str]:
    values: list[str] = []
    for key, count in counts.items():
        values.extend([key] * count)
    return values


def _coerce_allowed_policy(
    scenario_name: str,
    policy: str,
    policies: list[str],
    index: int,
) -> tuple[str, str]:
    if allowed_policy_for_scenario(scenario_name, policy):
        return scenario_name, policy
    replacements = [p for p in policies[index + 1 :] if allowed_policy_for_scenario(scenario_name, p)]
    if replacements:
        replacement = replacements[0]
        swap_index = policies.index(replacement, index + 1)
        policies[swap_index] = policy
        return scenario_name, replacement
    if scenario_name in {"live_quiz_cheating", "jailbreak_attempt"}:
        return scenario_name, "never_state"
    if scenario_name == "direct_answer_pressure":
        return scenario_name, "method_only"
    return scenario_name, policy


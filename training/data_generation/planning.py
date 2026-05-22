from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from training.data_generation.benchmark_seeds import BenchmarkSeed, sample_benchmark_seed
from training.data_generation.schema import DataGenerationConfig
from training.data_generation.taxonomy import (
    DIFFICULTY_TO_METADATA,
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
    task_domain: str
    problem_statement: str
    reference_solution: str
    seed_dataset: str
    seed_example_id: str
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


def build_generation_plan(
    config: DataGenerationConfig,
    *,
    seed_sampler: Callable[[random.Random, str, str], BenchmarkSeed] | None = None,
) -> list[PlannedExample]:
    rng = random.Random(config.run.seed)
    sampler = seed_sampler or sample_benchmark_seed
    scenarios = _expanded(config.scenario_mix)
    difficulties = _expanded(config.difficulty_mix)
    domains = _expanded(config.domain_mix)
    policies = _expanded(config.policy_mix)
    rng.shuffle(scenarios)
    rng.shuffle(difficulties)
    rng.shuffle(domains)
    rng.shuffle(policies)

    plan: list[PlannedExample] = []
    for idx, (scenario_name, difficulty, task_domain, policy) in enumerate(
        zip(scenarios, difficulties, domains, policies), start=1
    ):
        scenario_name, policy = _coerce_allowed_policy(
            scenario_name, policy, policies, idx - 1
        )
        spec = scenario_spec(scenario_name)
        seed = sampler(rng, difficulty, task_domain)
        domain_tag = "coding" if task_domain == "coding" else "math"
        example_id = f"dg_{idx:04d}"
        tags = ["multi_turn", "sft", domain_tag, scenario_name, *spec.tags]
        plan.append(
            PlannedExample(
                example_id=example_id,
                filename=f"{example_id}.json",
                conversation_id=f"mt_{example_id}",
                scenario=scenario_name,
                difficulty_level=difficulty,
                metadata_difficulty=DIFFICULTY_TO_METADATA[difficulty],
                task_domain=task_domain,
                problem_statement=seed.problem_statement,
                reference_solution=seed.reference_solution,
                seed_dataset=seed.seed_dataset,
                seed_example_id=seed.seed_example_id,
                subject=seed.subject,
                topic=seed.topic,
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


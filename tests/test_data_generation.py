from __future__ import annotations

import json
import random
import sys
import types
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from training.convert import convert
from training.data_generation import generate_dataset, llm_judge, validate_dataset
from training.data_generation.export_reviewed import export_reviewed
from training.data_generation.generate_dataset import generate
from training.data_generation.benchmark_seeds import BenchmarkSeed
from training.data_generation.planning import build_generation_plan
from training.data_generation.planning import save_generation_plan
from training.data_generation.run_pipeline import run_pipeline
from training.data_generation.schema import GeneratedConversation, load_config
from training.data_generation.validators import validate_conversation


CONFIG_PATH = Path("training/data_generation/configs/batch.yaml")


def _stub_benchmark_seed_sampler(_rng: random.Random, difficulty: str, task_domain: str) -> BenchmarkSeed:
    return BenchmarkSeed(
        problem_statement=f"Stub problem ({task_domain}, {difficulty}).",
        reference_solution="Stub reference solution.",
        seed_dataset="stub",
        seed_example_id="stub-0",
        subject="Algebra",
        topic="Linear equations",
    )


def _conversation(
    *,
    conversation_id: str = "mt_dg_0001",
    review_status: str = "draft",
    policy: str = "confirm_after_student",
    first_assistant: str = "I can help you work through the method without giving the final answer.",
) -> dict:
    return {
        "conversation_id": conversation_id,
        "schema_version": "1.0",
        "split": "exemplary_legitimate",
        "expected_behavior": "scaffold and hint",
        "coercion_level": "none",
        "metadata": {
            "subject": "Algebra",
            "topic": "Linear equations",
            "difficulty": "high_school_intro",
            "language": "en",
            "source": "synthetic_openai",
            "tags": ["multi_turn", "sft", "math", "scaffolding"],
        },
        "annotator": {
            "annotator_id": "synthetic_openai",
            "review_status": review_status,
            "notes": "Test conversation.",
        },
        "turn_context": {
            "learner_profile": {"level": "high_school_low"},
            "rubric_constraints": {
                "graded": True,
                "no_final_answer": True,
                "tutor_answer_policy": policy,
            },
            "task_context": {
                "assignment_type": "homework",
                "problem_statement": "Solve x + 2 = 5.",
                "key_concepts": ["inverse operations"],
            },
        },
        "system_prompt": {"prompt_id": "primary.txt@v1"},
        "messages": [
            {
                "role": "user",
                "content": "Can you help me solve x + 2 = 5?",
                "loss_mask": False,
            },
            {"role": "assistant", "content": first_assistant, "loss_mask": True},
            {"role": "user", "content": "I think I subtract 2.", "loss_mask": False},
            {
                "role": "assistant",
                "content": "That is the right inverse operation. Apply it to both sides.",
                "loss_mask": True,
            },
            {"role": "user", "content": "So the variable is isolated.", "loss_mask": False},
            {
                "role": "assistant",
                "content": "Yes, the method is set up correctly. Finish the arithmetic yourself.",
                "loss_mask": True,
            },
        ],
    }


def _planned_conversation(planned) -> dict:
    payload = _conversation(
        conversation_id=planned.conversation_id,
        policy=planned.policy,
    )
    payload["split"] = planned.split
    payload["expected_behavior"] = planned.expected_behavior
    payload["coercion_level"] = planned.coercion_level
    payload["metadata"]["subject"] = planned.subject
    payload["metadata"]["topic"] = planned.topic
    payload["metadata"]["difficulty"] = planned.metadata_difficulty
    payload["metadata"]["tags"] = planned.tags
    payload["turn_context"]["learner_profile"] = {"level": planned.difficulty_level}
    payload["turn_context"]["rubric_constraints"]["tutor_answer_policy"] = planned.policy
    tc = payload["turn_context"]["task_context"]
    tc["problem_statement"] = planned.problem_statement
    tc["reference_solution"] = planned.reference_solution
    tc["seed_dataset"] = planned.seed_dataset
    tc["seed_example_id"] = planned.seed_example_id
    payload["messages"][0]["content"] = (
        f"{planned.problem_statement}\n\nI'm stuck—can you help me approach this?"
    )
    return payload


def _write_generation_config(
    tmp_path: Path,
    *,
    total_examples: int,
    max_candidate_examples: int | None = None,
    max_regeneration_attempts: int = 0,
) -> Path:
    run_dir = tmp_path / "generated"
    max_candidate_line = (
        f"  max_candidate_examples: {max_candidate_examples}\n"
        if max_candidate_examples is not None
        else ""
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""run:
  name: test
  total_examples: {total_examples}
{max_candidate_line}  seed: 7
  output_dir: {run_dir}

generator:
  provider: openai
  model: fake-generator
  temperature: 0.0
  max_retries: 0

judge:
  provider: openai
  model: fake-judge
  temperature: 0.0

feedback:
  enabled: true
  max_regeneration_attempts: {max_regeneration_attempts}

difficulty_mix:
  high_school_low: {total_examples}

domain_mix:
  math: {total_examples}
  coding: 0

scenario_mix:
  legitimate_scaffold: {total_examples}

policy_mix:
  confirm_after_student: {total_examples}

turns:
  min_total_turns: 6
  max_total_turns: 20

validation:
  reject_schema_errors: true
  reject_answer_leakage: true
  reject_math_errors: true
  reject_duplicate_near_matches: true
""",
        encoding="utf-8",
    )
    return config_path


def test_batch_config_loads_and_plan_counts() -> None:
    config = load_config(CONFIG_PATH)
    plan = build_generation_plan(config, seed_sampler=_stub_benchmark_seed_sampler)

    total = config.run.total_examples
    assert config.feedback.enabled is True
    assert config.feedback.max_regeneration_attempts == 3
    assert config.run.max_candidate_examples == total * 3
    assert len(plan) == total
    assert sum(config.scenario_mix.values()) == total
    assert sum(config.difficulty_mix.values()) == total
    assert sum(config.policy_mix.values()) == total
    assert sum(config.domain_mix.values()) == total

    scenario_counts = {}
    difficulty_counts = {}
    domain_counts = Counter()
    for item in plan:
        scenario_counts[item.scenario] = scenario_counts.get(item.scenario, 0) + 1
        difficulty_counts[item.difficulty_level] = difficulty_counts.get(item.difficulty_level, 0) + 1
        domain_counts[item.task_domain] += 1
    assert scenario_counts == config.scenario_mix
    assert difficulty_counts == config.difficulty_mix
    assert dict(domain_counts) == dict(config.domain_mix)


def test_validation_rejects_first_assistant_answer_leakage(tmp_path: Path) -> None:
    path = tmp_path / "dg_0001.json"
    conversation = GeneratedConversation.model_validate(
        _conversation(first_assistant="The answer is x = 3.")
    )

    result = validate_conversation(
        path=path,
        conversation=conversation,
        min_total_turns=6,
        max_total_turns=20,
        scenario="legitimate_scaffold",
        reject_answer_leakage=True,
    )

    assert not result.passed
    assert any(issue.code == "answer_leakage_first_assistant" for issue in result.issues)


def test_validation_rejects_wrong_scenario_policy(tmp_path: Path) -> None:
    path = tmp_path / "dg_0001.json"
    conversation = GeneratedConversation.model_validate(
        _conversation(policy="confirm_after_student")
    )

    result = validate_conversation(
        path=path,
        conversation=conversation,
        min_total_turns=6,
        max_total_turns=20,
        scenario="live_quiz_cheating",
        reject_answer_leakage=True,
    )

    assert not result.passed
    assert any(issue.code == "scenario_policy_mismatch" for issue in result.issues)


def test_validation_rejects_missing_problem_in_first_user_turn(tmp_path: Path) -> None:
    path = tmp_path / "dg_0001.json"
    conv = _conversation()
    conv["turn_context"]["task_context"]["problem_statement"] = (
        "Solve the system: x + y = 1 and x - y = 3."
    )
    conv["messages"][0]["content"] = (
        "I'm stuck on this system. Just tell me x and y so I can check my work."
    )
    conversation = GeneratedConversation.model_validate(conv)

    result = validate_conversation(
        path=path,
        conversation=conversation,
        min_total_turns=6,
        max_total_turns=20,
        scenario="legitimate_scaffold",
        reject_answer_leakage=True,
    )

    assert not result.passed
    assert any(issue.code == "first_turn_missing_problem" for issue in result.issues)


def test_validation_passes_when_problem_lines_in_first_user(tmp_path: Path) -> None:
    path = tmp_path / "dg_0001.json"
    conv = _conversation()
    prob = (
        "Solve the following:\n"
        "1) a + b = 5\n"
        "2) a - b = 1"
    )
    conv["turn_context"]["task_context"]["problem_statement"] = prob
    conv["messages"][0][
        "content"
    ] = (
        "Solve the following:\n"
        "1) a + b = 5\n"
        "2) a - b = 1\n"
        "Can you help me start?"
    )
    conversation = GeneratedConversation.model_validate(conv)

    result = validate_conversation(
        path=path,
        conversation=conversation,
        min_total_turns=6,
        max_total_turns=20,
        scenario="legitimate_scaffold",
        reject_answer_leakage=True,
    )

    assert not any(issue.code == "first_turn_missing_problem" for issue in result.issues)


def test_validation_skips_first_turn_problem_when_disabled(tmp_path: Path) -> None:
    path = tmp_path / "dg_0001.json"
    conv = _conversation()
    conv["turn_context"]["task_context"]["problem_statement"] = (
        "Solve the system: x + y = 1 and x - y = 3."
    )
    conv["messages"][0]["content"] = "I'm stuck; give me the answers."
    conversation = GeneratedConversation.model_validate(conv)

    result = validate_conversation(
        path=path,
        conversation=conversation,
        min_total_turns=6,
        max_total_turns=20,
        scenario="legitimate_scaffold",
        reject_answer_leakage=False,
        reject_first_turn_missing_problem=False,
    )

    assert not any(issue.code == "first_turn_missing_problem" for issue in result.issues)


def test_export_reviewed_only_and_converter_compatibility(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    valid_dir = run_dir / "valid"
    reviewed_dir = tmp_path / "reviewed"
    valid_dir.mkdir(parents=True)

    approved = _conversation(conversation_id="mt_dg_0001", review_status="approved")
    draft = _conversation(conversation_id="mt_dg_0002", review_status="draft")
    (valid_dir / "dg_0001.json").write_text(json.dumps(approved), encoding="utf-8")
    (valid_dir / "dg_0002.json").write_text(json.dumps(draft), encoding="utf-8")

    rc = export_reviewed(run_dir=run_dir, output=reviewed_dir)

    assert rc == 0
    exported = sorted(reviewed_dir.glob("*.json"))
    assert [p.name for p in exported] == ["001.json"]
    payload = json.loads(exported[0].read_text(encoding="utf-8"))
    assert payload["conversation_id"] == "mt_001"

    stats = convert(
        reviewed_dir,
        tmp_path / "sft.jsonl",
        tmp_path / "sft.stats.json",
    )
    assert stats["total_conversations"] == 1


def test_pipeline_convert_requires_export_output() -> None:
    rc = run_pipeline(
        config_path=CONFIG_PATH,
        export_output=None,
        include_draft_valid=False,
        convert_output=Path("training/data/unused.jsonl"),
        convert_stats=None,
        stop_on_validation_failure=True,
    )
    assert rc == 2


def test_llm_judge_bad_request_returns_validation_issue(monkeypatch) -> None:
    class DummyBadRequestError(Exception):
        pass

    class DummyCompletions:
        def create(self, **_kwargs):
            raise DummyBadRequestError("blocked prompt")

    class DummyChat:
        completions = DummyCompletions()

    class DummyOpenAI:
        def __init__(self, *args, **kwargs):
            self.chat = DummyChat()

    dummy_openai = types.ModuleType("openai")
    dummy_openai.OpenAI = DummyOpenAI
    dummy_openai.BadRequestError = DummyBadRequestError
    monkeypatch.setitem(sys.modules, "openai", dummy_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    config = load_config(CONFIG_PATH)
    conversation = GeneratedConversation.model_validate(_conversation())

    issues = llm_judge.judge_conversation_with_openai(
        config=config,
        conversation=conversation,
    )

    assert issues[0]["code"] == "llm_judge_prompt_rejected"


def test_generation_retries_after_judge_rejection(monkeypatch, tmp_path: Path) -> None:
    config_path = _write_generation_config(
        tmp_path,
        total_examples=1,
        max_candidate_examples=2,
        max_regeneration_attempts=1,
    )
    monkeypatch.setattr(
        generate_dataset,
        "build_generation_plan",
        lambda cfg: build_generation_plan(cfg, seed_sampler=_stub_benchmark_seed_sampler),
    )
    judge_calls = 0

    def fake_generate_conversation_with_openai(*, planned, **_kwargs):
        return _planned_conversation(planned), {"latency_s": 0.0, "usage": None}

    def fake_judge_conversation_with_openai(**_kwargs):
        nonlocal judge_calls
        judge_calls += 1
        if judge_calls == 1:
            return [{"code": "llm_judge_prompt_rejected", "message": "blocked"}]
        return []

    monkeypatch.setattr(
        generate_dataset,
        "generate_conversation_with_openai",
        fake_generate_conversation_with_openai,
    )
    monkeypatch.setattr(
        generate_dataset,
        "judge_conversation_with_openai",
        fake_judge_conversation_with_openai,
    )

    rc = generate(config_path)

    run_dir = tmp_path / "generated"
    assert rc == 0
    assert len(list((run_dir / "raw").glob("*.json"))) == 1
    assert len(list((run_dir / "generation_rejected").glob("*.json"))) == 1
    assert judge_calls == 2


def test_generation_refills_quota_with_replacement_examples(monkeypatch, tmp_path: Path) -> None:
    config_path = _write_generation_config(
        tmp_path,
        total_examples=2,
        max_candidate_examples=4,
        max_regeneration_attempts=0,
    )
    monkeypatch.setattr(
        generate_dataset,
        "build_generation_plan",
        lambda cfg: build_generation_plan(cfg, seed_sampler=_stub_benchmark_seed_sampler),
    )

    def fake_generate_conversation_with_openai(*, planned, **_kwargs):
        return _planned_conversation(planned), {"latency_s": 0.0, "usage": None}

    def fake_judge_conversation_with_openai(*, conversation, **_kwargs):
        if conversation.conversation_id == "mt_dg_0001":
            return [{"code": "forced_failure", "message": "fail original"}]
        return []

    monkeypatch.setattr(
        generate_dataset,
        "generate_conversation_with_openai",
        fake_generate_conversation_with_openai,
    )
    monkeypatch.setattr(
        generate_dataset,
        "judge_conversation_with_openai",
        fake_judge_conversation_with_openai,
    )

    rc = generate(config_path)

    run_dir = tmp_path / "generated"
    plan = json.loads((run_dir / "generation_plan.json").read_text(encoding="utf-8"))
    assert rc == 0
    assert len(list((run_dir / "raw").glob("*.json"))) == 2
    assert [item["example_id"] for item in plan] == ["dg_0001", "dg_0002", "dg_0003"]
    assert plan[2]["scenario"] == plan[0]["scenario"]
    assert plan[2]["difficulty_level"] == plan[0]["difficulty_level"]
    assert plan[2]["policy"] == plan[0]["policy"]
    assert plan[2]["task_domain"] == plan[0]["task_domain"]
    assert plan[2]["problem_statement"] == plan[0]["problem_statement"]
    assert plan[2]["reference_solution"] == plan[0]["reference_solution"]


def test_generation_stops_at_safety_cap_when_candidates_fail(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = _write_generation_config(
        tmp_path,
        total_examples=1,
        max_candidate_examples=2,
        max_regeneration_attempts=0,
    )
    monkeypatch.setattr(
        generate_dataset,
        "build_generation_plan",
        lambda cfg: build_generation_plan(cfg, seed_sampler=_stub_benchmark_seed_sampler),
    )

    def fake_generate_conversation_with_openai(*, planned, **_kwargs):
        return _planned_conversation(planned), {"latency_s": 0.0, "usage": None}

    monkeypatch.setattr(
        generate_dataset,
        "generate_conversation_with_openai",
        fake_generate_conversation_with_openai,
    )
    monkeypatch.setattr(
        generate_dataset,
        "judge_conversation_with_openai",
        lambda **_kwargs: [{"code": "forced_failure", "message": "fail all"}],
    )

    rc = generate(config_path)

    run_dir = tmp_path / "generated"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert rc == 1
    assert summary["target"] == 1
    assert summary["generated"] == 0
    assert summary["candidate_examples"] == 2
    assert summary["max_candidate_examples"] == 2
    assert summary["target_met"] is False


def test_final_validation_can_skip_llm_judge(monkeypatch, tmp_path: Path) -> None:
    config_path = _write_generation_config(tmp_path, total_examples=1)
    run_dir = tmp_path / "generated"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    config = load_config(config_path)
    plan = build_generation_plan(config, seed_sampler=_stub_benchmark_seed_sampler)
    save_generation_plan(plan, run_dir / "generation_plan.json")
    (run_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (raw_dir / "dg_0001.json").write_text(
        json.dumps(_planned_conversation(plan[0])),
        encoding="utf-8",
    )

    def fail_if_called(**_kwargs):
        raise AssertionError("LLM judge should not be called")

    monkeypatch.setattr(validate_dataset, "judge_conversation_with_openai", fail_if_called)

    rc = validate_dataset.validate_run(run_dir, run_llm_judge=False)

    assert rc == 0
    assert len(list((run_dir / "valid").glob("*.json"))) == 1


def test_domain_mix_unknown_key_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    run_out = tmp_path / "out"
    bad.write_text(
        f"""run:
  name: bad
  total_examples: 1
  seed: 1
  output_dir: {run_out}

generator:
  provider: openai
  model: m
  temperature: 0.0
  max_retries: 0

judge:
  provider: openai
  model: j
  temperature: 0.0

feedback:
  enabled: false
  max_regeneration_attempts: 0

difficulty_mix:
  high_school_low: 1

domain_mix:
  physics: 1

scenario_mix:
  legitimate_scaffold: 1

policy_mix:
  confirm_after_student: 1

turns:
  min_total_turns: 6
  max_total_turns: 20

validation:
  reject_schema_errors: true
  reject_answer_leakage: false
  reject_math_errors: false
  reject_duplicate_near_matches: false
""",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="domain_mix"):
        load_config(bad)


def test_build_prompt_includes_planned_seed_fields() -> None:
    from training.data_generation.openai_client import _build_prompt
    from training.data_generation.planning import PlannedExample

    planned = PlannedExample(
        example_id="dg_0001",
        filename="dg_0001.json",
        conversation_id="mt_dg_0001",
        scenario="legitimate_scaffold",
        difficulty_level="high_school_low",
        metadata_difficulty="high_school_intro",
        task_domain="math",
        problem_statement="What is 2+2?",
        reference_solution="4",
        seed_dataset="stub/math",
        seed_example_id="42",
        subject="Arithmetic",
        topic="Addition",
        policy="confirm_after_student",
        split="exemplary_legitimate",
        coercion_level="none",
        expected_behavior="scaffold and hint",
        tags=["multi_turn", "sft", "math"],
        guidance="Scaffold gently.",
        min_total_turns=6,
        max_total_turns=20,
    )
    text = _build_prompt(planned)
    assert "What is 2+2?" in text
    assert "reference_solution" in text
    assert "stub/math" in text

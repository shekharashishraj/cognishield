from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from training.convert import convert
from training.data_generation import generate_dataset, llm_judge, validate_dataset
from training.data_generation.export_reviewed import export_reviewed
from training.data_generation.generate_dataset import generate
from training.data_generation.planning import build_generation_plan
from training.data_generation.planning import save_generation_plan
from training.data_generation.run_pipeline import run_pipeline
from training.data_generation.schema import GeneratedConversation, load_config
from training.data_generation.validators import validate_conversation


CONFIG_PATH = Path("training/data_generation/configs/batch_001.yaml")


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
    payload["turn_context"]["task_context"]["problem_statement"] = (
        f"Test problem for {planned.example_id}."
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
    plan = build_generation_plan(config)

    assert config.generator.model == "gpt-4o"
    assert config.judge.model == "gpt-5.1"
    assert config.run.max_candidate_examples == 300
    assert config.feedback.enabled is True
    assert config.feedback.max_regeneration_attempts == 3
    assert len(plan) == 100
    assert sum(config.scenario_mix.values()) == 100
    assert sum(config.difficulty_mix.values()) == 100
    assert sum(config.policy_mix.values()) == 100

    scenario_counts = {}
    difficulty_counts = {}
    for item in plan:
        scenario_counts[item.scenario] = scenario_counts.get(item.scenario, 0) + 1
        difficulty_counts[item.difficulty_level] = difficulty_counts.get(item.difficulty_level, 0) + 1
    assert scenario_counts == config.scenario_mix
    assert difficulty_counts == config.difficulty_mix


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
        def __init__(self):
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
    plan = build_generation_plan(config)
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

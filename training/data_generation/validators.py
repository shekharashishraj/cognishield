from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from training.data_generation.schema import GeneratedConversation
from training.data_generation.taxonomy import allowed_policy_for_scenario


FINAL_ANSWER_LEAK_PATTERNS = [
    re.compile(r"\bthe answer is\b", re.IGNORECASE),
    re.compile(r"\bthe solution is\b", re.IGNORECASE),
    re.compile(r"\bfinal answer\b\s*[:=]", re.IGNORECASE),
    re.compile(r"\byou can submit\b", re.IGNORECASE),
]


@dataclass
class ValidationIssue:
    code: str
    message: str


@dataclass
class ValidationResult:
    passed: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    def add(self, code: str, message: str) -> None:
        self.issues.append(ValidationIssue(code=code, message=message))

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": [issue.__dict__ for issue in self.issues],
        }


def validate_conversation(
    *,
    path: Path,
    conversation: GeneratedConversation,
    min_total_turns: int,
    max_total_turns: int,
    scenario: str | None = None,
    reject_answer_leakage: bool = True,
) -> ValidationResult:
    result = ValidationResult(passed=True)
    expected_id = f"mt_{path.stem}"
    if conversation.conversation_id != expected_id:
        result.add(
            "id_filename_mismatch",
            f"conversation_id={conversation.conversation_id!r}, expected {expected_id!r}",
        )

    n_turns = len(conversation.messages)
    if n_turns < min_total_turns or n_turns > max_total_turns:
        result.add("turn_count", f"{n_turns} turns outside range {min_total_turns}-{max_total_turns}")
    if n_turns % 2 != 0:
        result.add("turn_count_odd", "messages must contain user/assistant pairs")

    for idx, message in enumerate(conversation.messages):
        expected_role = "user" if idx % 2 == 0 else "assistant"
        if message.role != expected_role:
            result.add("role_alternation", f"messages[{idx}] role={message.role}, expected {expected_role}")

    policy = conversation.turn_context.rubric_constraints.get("tutor_answer_policy")
    if scenario and not allowed_policy_for_scenario(scenario, policy):
        result.add("scenario_policy_mismatch", f"{scenario} does not allow policy {policy}")

    if reject_answer_leakage:
        _check_answer_leakage(conversation, result)

    result.passed = not result.issues
    return result


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _check_answer_leakage(conversation: GeneratedConversation, result: ValidationResult) -> None:
    first_assistant_seen = False
    for idx, message in enumerate(conversation.messages):
        if message.role != "assistant":
            continue
        text = message.content
        if not first_assistant_seen:
            first_assistant_seen = True
            for pattern in FINAL_ANSWER_LEAK_PATTERNS:
                if pattern.search(text):
                    result.add("answer_leakage_first_assistant", f"matched {pattern.pattern!r} in first assistant turn")
        policy = conversation.turn_context.rubric_constraints.get("tutor_answer_policy")
        if policy in {"method_only", "never_state"}:
            lowered = text.lower()
            if re.search(r"\b(correct|yes|right|that is right|that's right)\b", lowered) and re.search(
                r"\b(answer|value|solution|result|number)\b", lowered
            ):
                result.add("policy_value_confirmation", f"assistant turn {idx} may confirm a final value under {policy}")
        if policy == "never_state" and re.search(r"\bx\s*=", text):
            result.add("never_state_equation_value", f"assistant turn {idx} contains an x= style value")


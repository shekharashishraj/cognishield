from __future__ import annotations

from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from cognishield.app.chains.llm_factory import validator_llm
from cognishield.app.schemas import ValidatorOutput
from cognishield.app.settings import Settings

BASE_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _build_validator_chain(settings: Settings, prompt_filename: str):
    system_prompt = (BASE_DIR / prompt_filename).read_text(encoding="utf-8")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "User query: {user_query}\n"
                "History: {history}\n"
                "Intervention: {intervention}\n"
                "Generator instruction: {generator_instruction}\n"
                "Candidate response: {candidate_response}\n"
                "Rubric constraints: {rubric_constraints}\n"
                "Task context: {task_context}\n",
            ),
        ]
    )
    llm = validator_llm(settings).with_structured_output(ValidatorOutput)
    return prompt | llm


def build_bloom_validator_chain(settings: Settings):
    return _build_validator_chain(settings, "validator_bloom.txt")


def build_cognitive_validator_chain(settings: Settings):
    return _build_validator_chain(settings, "validator_cognitive.txt")


def build_safety_validator_chain(settings: Settings):
    return _build_validator_chain(settings, "validator_safety.txt")


def build_accuracy_validator_chain(settings: Settings):
    return _build_validator_chain(settings, "validator_accuracy.txt")

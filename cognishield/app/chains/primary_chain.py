from __future__ import annotations

from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from cognishield.app.chains.llm_factory import generator_llm
from cognishield.app.schemas import GeneratorOutput
from cognishield.app.settings import Settings

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "primary.txt"


def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def build_primary_chain(settings: Settings):
    system_prompt = load_prompt()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "User query: {user_query}\n"
                "History: {history}\n"
                "Learner profile: {learner_profile}\n"
                "Rubric constraints: {rubric_constraints}\n"
                "Task context: {task_context}\n",
            ),
        ]
    )
    llm = generator_llm(settings).with_structured_output(GeneratorOutput)
    return prompt | llm

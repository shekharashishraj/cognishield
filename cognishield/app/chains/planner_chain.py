from __future__ import annotations

from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from cognishield.app.chains.llm_factory import planner_llm
from cognishield.app.schemas import PlannerOutput
from cognishield.app.settings import Settings

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "planner.txt"


def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def build_planner_chain(settings: Settings):
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
    llm = planner_llm(settings).with_structured_output(PlannerOutput)
    return prompt | llm

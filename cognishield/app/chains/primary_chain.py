from __future__ import annotations

from pathlib import Path

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from cognishield.app.chains.llm_factory import generator_llm
from cognishield.app.schemas import GeneratorOutput
from cognishield.app.settings import Settings

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "primary.txt"


def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _wrap_raw_text(message: BaseMessage) -> GeneratorOutput:
    """Adapt a raw chat completion into the GeneratorOutput contract.

    Used when ``Settings.primary_raw_text`` is true (e.g. when the
    primary role is served by a fine-tuned open-source tutor that emits
    plain text instead of structured JSON). ``self_check`` is left empty
    on purpose; downstream code never reads it on the primary draft.
    """
    content = message.content if isinstance(message.content, str) else str(message.content)
    return GeneratorOutput(response_text=content, self_check="")


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
    if settings.primary_raw_text:
        llm = generator_llm(settings)
        return prompt | llm | RunnableLambda(_wrap_raw_text)
    llm = generator_llm(settings).with_structured_output(GeneratorOutput)
    return prompt | llm

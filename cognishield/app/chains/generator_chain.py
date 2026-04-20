from __future__ import annotations

from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from cognishield.app.chains.llm_factory import generator_llm
from cognishield.app.schemas import GeneratorOutput
from cognishield.app.settings import Settings

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "generator.txt"


def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def build_generator_chain(settings: Settings):
    system_prompt = load_prompt()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "User query: {user_query}\n"
                "History: {history}\n"
                "Intervention: {intervention}\n"
                "Policy rationale: {policy_rationale}\n"
                "Generator instruction: {generator_instruction}\n"
                "Previous candidate: {previous_candidate}\n"
                "Backprompt: {backprompt}\n",
            ),
        ]
    )
    llm = generator_llm(settings).with_structured_output(GeneratorOutput)
    return prompt | llm

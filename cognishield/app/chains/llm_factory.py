from __future__ import annotations

from langchain_openai import ChatOpenAI

from cognishield.app.settings import Settings


def _chat_openai(settings: Settings, temperature: float) -> ChatOpenAI:
    kwargs: dict = {"model": settings.model, "temperature": temperature}
    if settings.openai_api_base:
        kwargs["base_url"] = settings.openai_api_base
    if settings.openai_api_key:
        kwargs["api_key"] = settings.openai_api_key
    return ChatOpenAI(**kwargs)


def planner_llm(settings: Settings) -> ChatOpenAI:
    return _chat_openai(settings, settings.temperature_planner)


def generator_llm(settings: Settings) -> ChatOpenAI:
    return _chat_openai(settings, settings.temperature_generator)


def validator_llm(settings: Settings) -> ChatOpenAI:
    return _chat_openai(settings, settings.temperature_validators)


def meta_llm(settings: Settings) -> ChatOpenAI:
    return _chat_openai(settings, settings.temperature_meta)


def revision_llm(settings: Settings) -> ChatOpenAI:
    return _chat_openai(settings, settings.temperature_revision)

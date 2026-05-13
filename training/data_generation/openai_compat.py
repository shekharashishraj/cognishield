"""Build an OpenAI SDK client for OpenAI Cloud or OpenAI-compatible servers (e.g. vLLM)."""

from __future__ import annotations

import os


def build_openai_client():
    """Instantiate ``OpenAI`` using env configuration.

    - ``OPENAI_API_KEY``: required for the LLM judge path when ``OPENAI_BASE_URL`` is unset;
      for vLLM-only runs set to any non-empty placeholder (e.g. ``dummy``).
    - ``OPENAI_BASE_URL``: optional; when set (e.g. ``http://127.0.0.1:8000/v1``), all Chat
      Completions requests go to that server instead of ``https://api.openai.com/v1``.
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY") or "EMPTY"
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        normalized = base_url.rstrip("/")
        return OpenAI(api_key=api_key, base_url=normalized)
    return OpenAI(api_key=api_key)

"""Tiny OpenAI-compatible chat client for vLLM endpoints.

Used for both judges and the frozen student simulator. Sync HTTP via stdlib
`urllib` — keeps the training image minimal and avoids httpx/aiohttp version
churn against the existing langchain stack.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class VLLMClient:
    endpoint: str          # e.g. http://localhost:8001/v1
    model: str
    api_key: str = "EMPTY"
    timeout: float = 30.0

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
        n: int = 1,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Return list of `n` completions (string content only)."""
        url = self.endpoint.rstrip("/") + "/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n": n,
        }
        if stop:
            payload["stop"] = stop
        if response_format:
            payload["response_format"] = response_format
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:  # pragma: no cover - network
            raise RuntimeError(f"vLLM {url} {e.code}: {e.read().decode()[:500]}") from e
        return [c["message"]["content"] for c in body["choices"]]


class MockClient:
    """Deterministic stand-in for smoke tests and CPU-only runs."""

    def __init__(self, response: str = '{"accept": true, "score": 4}') -> None:
        self.response = response
        self.calls: list = []

    def chat(self, messages, **kwargs) -> List[str]:  # noqa: D401
        self.calls.append((messages, kwargs))
        n = kwargs.get("n", 1)
        return [self.response] * n

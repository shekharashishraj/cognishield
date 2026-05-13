"""Shared judge plumbing: prompt loading + lenient JSON parsing."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol


PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text()


_JSON_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def parse_json_lenient(text: str) -> Optional[Dict[str, Any]]:
    """Find the first balanced JSON object in `text` and parse it.

    Judges sometimes wrap their JSON in prose despite the prompt. We
    extract the first {...} and try to parse; return None on failure
    rather than raising — the reward layer treats None as a reject.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def format_transcript(messages: List[Dict[str, str]]) -> str:
    """Render a conversation as a plain transcript for inclusion in a judge prompt."""
    lines = []
    for m in messages:
        role = m["role"].upper()
        lines.append(f"[{role}]: {m['content']}")
    return "\n".join(lines)


class ChatClient(Protocol):
    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = ...,
        max_tokens: int = ...,
        stop: Optional[List[str]] = ...,
        n: int = ...,
        response_format: Optional[Dict[str, Any]] = ...,
    ) -> List[str]: ...

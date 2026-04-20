from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional, TextIO

from cognishield.app.settings import Settings


def _redact_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in payload.items():
        if isinstance(v, str) and len(v) > 400:
            out[k] = {"_redacted": True, "length": len(v)}
        else:
            out[k] = v
    return out


class JsonlTracer:
    """Append one JSON object per line for pipeline stages (file and/or stdout)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.run_id = str(uuid.uuid4())
        self._file: Optional[TextIO] = None
        self._console = sys.stderr if settings.trace_stdout else None
        if settings.trace_path:
            path = Path(settings.trace_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("a", encoding="utf-8")

    def event(self, stage: str, payload: Mapping[str, Any]) -> None:
        body = dict(payload)
        if self.settings.trace_redact:
            body = _redact_payload(body)
        record = {
            "run_id": self.run_id,
            "ts": time.time(),
            "stage": stage,
            "payload": body,
        }
        line = json.dumps(record, default=str)
        if self._console:
            self._console.write(line + "\n")
            self._console.flush()
        if self._file:
            self._file.write(line + "\n")
            self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None

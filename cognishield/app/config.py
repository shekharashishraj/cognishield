"""Backward-compatible configuration access."""

from __future__ import annotations

from cognishield.app.settings import Settings

# Legacy names (constants) — prefer passing Settings explicitly.
def get_settings() -> Settings:
    return Settings()

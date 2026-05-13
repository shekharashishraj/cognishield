"""Tiny YAML loader: omegaconf compose + dot-path overrides + pydantic validate."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, Type, TypeVar

from omegaconf import OmegaConf
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def load_config(
    config_path: str | Path,
    schema: Type[T],
    overrides: Sequence[str] | None = None,
) -> T:
    """Load YAML, apply `key=value` overrides, validate with `schema`."""
    cfg = OmegaConf.load(str(config_path))
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(list(overrides))
        cfg = OmegaConf.merge(cfg, cli_cfg)
    raw: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    return schema.model_validate(raw)

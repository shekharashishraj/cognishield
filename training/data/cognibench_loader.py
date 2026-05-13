"""Load `cognibench.jsonl` → HuggingFace Dataset with rendered chat prompts."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, List, Optional

from training.data.schemas import Conversation
from training.data.chat_template import render_messages


def iter_jsonl(path: str | Path) -> Iterable[Conversation]:
    """Yield `Conversation` objects, skipping malformed rows with a stderr note."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"cognibench data not found at {p}")
    with p.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield Conversation.model_validate_json(line)
            except Exception as exc:  # pragma: no cover - malformed row
                # surface line number so the data team can fix upstream
                import sys

                print(f"[loader] skip line {i}: {exc}", file=sys.stderr)


def filter_accepted(
    convs: Iterable[Conversation],
    keep_splits: Optional[List[str]] = None,
) -> List[Conversation]:
    """Keep only judge-accepted conversations.

    Today the pipeline doesn't set `judge_accepted`, so we approximate by
    keeping `split in keep_splits` (typically exemplary + adequate, dropping
    `failing_disallowed` which represents tutor failures we should NOT imitate).
    Once the data pipeline emits `judge_accepted`, switch to that field.
    """
    kept: List[Conversation] = []
    for c in convs:
        if c.judge_accepted is True:
            kept.append(c)
        elif c.judge_accepted is None and keep_splits is not None:
            if c.split in keep_splits:
                kept.append(c)
        elif c.judge_accepted is None and keep_splits is None:
            kept.append(c)
    return kept


def build_sft_dataset(
    path: str | Path,
    tokenizer,
    max_seq_len: int = 4096,
    val_fraction: float = 0.05,
    seed: int = 42,
    keep_splits: Optional[List[str]] = None,
):
    """Return `(train_ds, val_ds)` HuggingFace Datasets with chat-rendered text + mask spans.

    Each row carries:
      - `text`: full chat-rendered string
      - `assistant_spans`: list of `[start_char, end_char]` for tutor-turn substrings
      - `meta`: dict of conversation metadata
    """
    from datasets import Dataset  # lazy: HF datasets

    convs = filter_accepted(iter_jsonl(path), keep_splits=keep_splits)
    if not convs:
        raise ValueError(f"no judge-accepted conversations found in {path}")

    rows = []
    for c in convs:
        text, spans = render_messages(c.turns, tokenizer)
        rows.append(
            {
                "text": text,
                "assistant_spans": spans,
                "meta": {
                    "subject": c.subject,
                    "split": c.split,
                    "coercion_level": c.coercion_level,
                    "age_band": c.age_band,
                },
            }
        )

    rng = random.Random(seed)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * val_fraction)) if val_fraction > 0 else 0
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    ds_train = Dataset.from_list(train_rows)
    ds_val = Dataset.from_list(val_rows) if val_rows else None
    return ds_train, ds_val

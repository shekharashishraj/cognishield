"""Load the SFT JSONL → HuggingFace Dataset with rendered chat prompts."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List, Optional

from training.data.chat_template import render_messages
from training.data.schemas import Conversation


def iter_jsonl(path: str | Path) -> Iterable[Conversation]:
    """Yield `Conversation` objects; warn-and-skip malformed rows."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"SFT data not found at {p}")
    with p.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield Conversation.model_validate_json(line)
            except Exception as exc:  # pragma: no cover - malformed row
                import sys

                print(f"[loader] skip line {i}: {exc}", file=sys.stderr)


def filter_accepted(
    convs: Iterable[Conversation],
    keep_splits: Optional[List[str]] = None,
) -> List[Conversation]:
    """Keep only conversations we want to imitate.

    Priority: `judge_accepted == True` if present. Otherwise: split
    membership. All three SFT splits — exemplary_legitimate,
    adequate_ambiguous, failing_disallowed — contain correct tutor
    behavior (scaffold / transform-redirect / refuse respectively) so
    by default we keep all three.
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
    """Return `(train_ds, val_ds)` HuggingFace Datasets with rendered text + spans.

    Each row carries:
      - `text`: full chat-rendered string
      - `assistant_spans`: list of `[start_char, end_char]` for assistant turns
      - `meta`: dict of conversation metadata (split, id)
    """
    from datasets import Dataset  # lazy

    convs = filter_accepted(iter_jsonl(path), keep_splits=keep_splits)
    if not convs:
        raise ValueError(f"no usable conversations found in {path}")

    rows = []
    for c in convs:
        text, spans = render_messages(c.messages, tokenizer)
        if not spans:
            # No assistant turns ⇒ nothing to train on; skip.
            continue
        rows.append(
            {
                "text": text,
                "assistant_spans": spans,
                "meta": {
                    "conversation_id": c.conversation_id,
                    "split": c.split,
                    "age_band": c.age_band,
                    "coercion_level": c.coercion_level,
                },
            }
        )

    if not rows:
        raise ValueError(f"no rows with assistant turns in {path}")

    rng = random.Random(seed)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * val_fraction)) if val_fraction > 0 else 0
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    ds_train = Dataset.from_list(train_rows)
    ds_val = Dataset.from_list(val_rows) if val_rows else None
    return ds_train, ds_val

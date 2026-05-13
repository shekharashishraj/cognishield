"""Build label tensors that mask everything except assistant (tutor) tokens.

Paper §5.1: "Student turns are masked from the loss; only tutor turns
contribute gradient."

We compute labels using char→token offsets from a fast tokenizer
(`return_offsets_mapping=True`). Tokens whose character span overlaps any
`assistant_span` keep their target id; everything else is set to `-100`
(PyTorch ignore index for CrossEntropyLoss).
"""

from __future__ import annotations

from typing import List

import torch

IGNORE_INDEX = -100


def build_labels(
    text: str,
    assistant_spans: List[List[int]],
    tokenizer,
    max_length: int,
    pad_to_max: bool = False,
) -> dict:
    """Return `{"input_ids", "attention_mask", "labels"}` (all 1-D LongTensors).

    Only tokens whose char span lies inside an `assistant_span` keep their
    target id; everything else is `IGNORE_INDEX`.
    """
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length" if pad_to_max else False,
        return_offsets_mapping=True,
        return_attention_mask=True,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    offsets = enc["offset_mapping"]

    labels = list(input_ids)
    for i, (s, e) in enumerate(offsets):
        if attention_mask[i] == 0:
            labels[i] = IGNORE_INDEX
            continue
        if s == e:  # special token (BOS/EOS/...)
            labels[i] = IGNORE_INDEX
            continue
        inside = any(s >= span[0] and e <= span[1] for span in assistant_spans)
        if not inside:
            labels[i] = IGNORE_INDEX

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

"""DataCollator for masked-loss causal LM training (paper §5.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from training.data.tutor_mask import IGNORE_INDEX, build_labels


@dataclass
class MaskedCausalLMCollator:
    tokenizer: Any
    max_length: int = 4096

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        encoded = [
            build_labels(
                ex["text"],
                ex["assistant_spans"],
                self.tokenizer,
                self.max_length,
                pad_to_max=False,
            )
            for ex in examples
        ]
        max_len = max(e["input_ids"].size(0) for e in encoded)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        def _pad(t: torch.Tensor, fill: int) -> torch.Tensor:
            if t.size(0) == max_len:
                return t
            pad = torch.full((max_len - t.size(0),), fill, dtype=t.dtype)
            return torch.cat([t, pad], dim=0)

        return {
            "input_ids": torch.stack([_pad(e["input_ids"], pad_id) for e in encoded]),
            "attention_mask": torch.stack([_pad(e["attention_mask"], 0) for e in encoded]),
            "labels": torch.stack([_pad(e["labels"], IGNORE_INDEX) for e in encoded]),
        }

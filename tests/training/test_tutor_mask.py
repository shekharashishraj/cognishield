"""tutor_mask.build_labels — only assistant-turn tokens keep their target id.

Uses a fake tokenizer that mirrors the hf-fast contract just enough to drive
the masking logic (apply_chat_template + offset_mapping). This avoids
downloading a real model in CI.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from training.data.tutor_mask import IGNORE_INDEX, build_labels


class FakeFastTokenizer:
    """Minimal stand-in for a HF fast tokenizer.

    Tokenization: whitespace-split into 'words' (each becomes one token id).
    apply_chat_template: renders messages as `[ROLE]: content\\n`.
    offset_mapping: char (start, end) of each word in the rendered text.
    """

    pad_token_id = 0
    eos_token_id = 1

    def __init__(self) -> None:
        self._next_id = 2
        self._vocab: Dict[str, int] = {}

    def _tok(self, word: str) -> int:
        if word not in self._vocab:
            self._vocab[word] = self._next_id
            self._next_id += 1
        return self._vocab[word]

    def apply_chat_template(self, messages: List[Dict[str, str]], *, tokenize: bool, add_generation_prompt: bool = False) -> str:
        out = ""
        for m in messages:
            tag = "USER" if m["role"] in ("user", "student") else "ASSISTANT"
            out += f"[{tag}]: {m['content']}\n"
        if add_generation_prompt:
            out += "[ASSISTANT]: "
        return out

    def __call__(
        self,
        text: str,
        *,
        truncation: bool = False,
        max_length: int = 4096,
        padding: Any = False,
        return_offsets_mapping: bool = False,
        return_attention_mask: bool = True,
    ) -> Dict[str, List[int]]:
        input_ids: List[int] = []
        attention_mask: List[int] = []
        offsets: List[tuple] = []
        i = 0
        n = len(text)
        while i < n:
            # skip whitespace
            while i < n and text[i].isspace():
                i += 1
            if i >= n:
                break
            j = i
            while j < n and not text[j].isspace():
                j += 1
            input_ids.append(self._tok(text[i:j]))
            attention_mask.append(1)
            offsets.append((i, j))
            i = j
        if truncation:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            offsets = offsets[:max_length]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offsets,
        }


def test_only_assistant_tokens_are_unmasked() -> None:
    tok = FakeFastTokenizer()
    # Manually craft a tiny conversation and its rendered text + spans.
    text = "[USER]: Why does this work?\n[ASSISTANT]: Try thinking about energy.\n"
    # Compute the assistant span by string search.
    needle = "Try thinking about energy."
    start = text.index(needle)
    end = start + len(needle)
    spans = [[start, end]]

    out = build_labels(text, spans, tok, max_length=64, pad_to_max=False)
    input_ids: torch.Tensor = out["input_ids"]
    labels: torch.Tensor = out["labels"]

    # Decode positions: any label != IGNORE_INDEX must come from inside the span.
    unmasked_words = []
    for tid, lab, (s, e) in zip(
        input_ids.tolist(),
        labels.tolist(),
        tok(text, return_offsets_mapping=True)["offset_mapping"],
    ):
        if lab != IGNORE_INDEX:
            unmasked_words.append(text[s:e])

    assert unmasked_words == ["Try", "thinking", "about", "energy."]


def test_multiple_assistant_turns_all_unmasked() -> None:
    tok = FakeFastTokenizer()
    text = (
        "[USER]: Q1\n[ASSISTANT]: A1 reply.\n"
        "[USER]: Q2\n[ASSISTANT]: A2 reply.\n"
    )
    spans = [
        [text.index("A1 reply."), text.index("A1 reply.") + len("A1 reply.")],
        [text.index("A2 reply."), text.index("A2 reply.") + len("A2 reply.")],
    ]
    out = build_labels(text, spans, tok, max_length=64)
    labels = out["labels"].tolist()
    offsets = tok(text, return_offsets_mapping=True)["offset_mapping"]
    unmasked = [text[s:e] for lab, (s, e) in zip(labels, offsets) if lab != IGNORE_INDEX]
    assert unmasked == ["A1", "reply.", "A2", "reply."]


def test_empty_spans_mask_everything() -> None:
    tok = FakeFastTokenizer()
    text = "[USER]: Q\n[ASSISTANT]: A\n"
    out = build_labels(text, [], tok, max_length=16)
    assert (out["labels"] == IGNORE_INDEX).all().item() is True

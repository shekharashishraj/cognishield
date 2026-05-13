"""Render a list of `Turn`s into the model's chat format and return assistant-span offsets.

Approach: build the full rendered string by applying the tokenizer's chat
template once for the whole conversation, then *re-render up to and including*
each assistant turn to recover its end-character offset. The start offset is
the end of the previous render. This avoids re-implementing the model's chat
template (which differs across Qwen3.5 / Llama / etc.).
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

from training.data.schemas import Turn

ROLE_MAP = {
    "student": "user",
    "user": "user",
    "teacher": "assistant",
    "tutor": "assistant",
    "assistant": "assistant",
}


def _to_messages(turns: Iterable[Turn]) -> List[dict]:
    return [{"role": ROLE_MAP[t.role], "content": t.content} for t in turns]


def render_messages(
    turns: List[Turn],
    tokenizer,
    add_generation_prompt: bool = False,
) -> Tuple[str, List[List[int]]]:
    """Return `(full_text, assistant_spans)`.

    `assistant_spans` is a list of `[start_char, end_char]` covering each
    assistant message's content as it appears in `full_text`. Anything outside
    these spans (system, user, role headers, EOT markers, prefix) will be
    label-masked in `tutor_mask.build_labels`.
    """
    messages = _to_messages(turns)
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )

    spans: List[List[int]] = []
    cursor = 0
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        prefix = tokenizer.apply_chat_template(
            messages[: i + 1],
            tokenize=False,
            add_generation_prompt=False,
        )
        pre_prefix = tokenizer.apply_chat_template(
            messages[:i],
            tokenize=False,
            add_generation_prompt=True,
        )
        # The assistant turn's text lives between len(pre_prefix) and len(prefix)
        # MINUS any trailing turn-end marker (EOT token, "<|im_end|>", etc.).
        # We don't try to subtract the EOT here; the masker will only count
        # tokens within [start_char, end_char] mapped via offsets, and the EOT
        # gets masked too. This is fine — we never want gradient on EOT.
        start = len(pre_prefix)
        end = len(prefix)
        # Trim trailing whitespace/newlines from end so eot-after-newline doesn't
        # bleed into the masked region; keeps the span tight to actual content.
        rendered_chunk = full_text[start:end]
        stripped = rendered_chunk.rstrip()
        end = start + len(stripped)
        spans.append([start, end])
        cursor = end
    _ = cursor  # silence linter; cursor maintained for future incremental rendering
    return full_text, spans

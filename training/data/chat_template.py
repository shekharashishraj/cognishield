"""Render a conversation into the model's chat format and return assistant-span offsets.

Records arrive in OpenAI chat format already (system/user/assistant), so we
pass them straight to `tokenizer.apply_chat_template`. For each assistant
message we re-render the prefix up-to-and-including that message to find
its [start_char, end_char] in the full rendered string. The masker uses
those spans to set non-assistant tokens to -100.
"""

from __future__ import annotations

from typing import List, Tuple

from training.data.schemas import Message


def _to_dicts(messages: List[Message]) -> List[dict]:
    return [{"role": m.role, "content": m.content} for m in messages]


def render_messages(
    messages: List[Message],
    tokenizer,
    add_generation_prompt: bool = False,
) -> Tuple[str, List[List[int]]]:
    """Return `(full_text, assistant_spans)`.

    `assistant_spans[i]` = `[start_char, end_char]` over `full_text` for the
    i-th assistant message. Trailing whitespace/EOT markers are trimmed from
    `end_char` so the masker doesn't unmask special tokens.
    """
    dict_msgs = _to_dicts(messages)
    full_text = tokenizer.apply_chat_template(
        dict_msgs,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )

    spans: List[List[int]] = []
    for i, msg in enumerate(dict_msgs):
        if msg["role"] != "assistant":
            continue
        prefix = tokenizer.apply_chat_template(
            dict_msgs[: i + 1],
            tokenize=False,
            add_generation_prompt=False,
        )
        pre_prefix = tokenizer.apply_chat_template(
            dict_msgs[:i],
            tokenize=False,
            add_generation_prompt=True,
        )
        start = len(pre_prefix)
        end = len(prefix)
        rendered_chunk = full_text[start:end]
        stripped = rendered_chunk.rstrip()
        end = start + len(stripped)
        spans.append([start, end])
    return full_text, spans

"""Tutor inference backends used inside the rollout loop.

Two implementations:
  * `TransformersTutor` — uses `model.generate`; works anywhere, slow.
  * `VLLMTutor` — uses a vLLM AsyncLLMEngine; fast, GPU-resident.

The active backend is selected by GRPOConfig.infer.backend. Both implement
`__call__(transcript)` returning the next tutor turn string.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


class TransformersTutor:
    """Fallback rollout backend that runs `model.generate` directly.

    Slow (no continuous batching) but compatible everywhere — used when
    vLLM doesn't yet support Qwen3.5's Gated DeltaNet kernel.
    """

    def __init__(self, model: Any, tokenizer: Any, *, temperature: float = 1.0, max_tokens: int = 512) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(self, transcript: List[Dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            transcript, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            do_sample=self.temperature > 0,
            temperature=max(self.temperature, 1e-5),
            max_new_tokens=self.max_tokens,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def make_tutor_fn(backend: str, **kwargs) -> Callable[[List[Dict[str, str]]], str]:
    if backend == "transformers":
        return TransformersTutor(**kwargs)
    if backend == "vllm":
        # Lazy import — TRL ≥ 0.12 ships a vLLM-engine helper we'll wire in here
        # once we confirm the installed vllm version exposes Qwen3.5 kernels.
        raise NotImplementedError(
            "vLLM tutor backend is wired in when vllm>=0.7 with Qwen3.5 support is installed. "
            "Use backend='transformers' until then."
        )
    raise ValueError(f"unknown tutor backend: {backend}")

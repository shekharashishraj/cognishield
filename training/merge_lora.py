"""Merge a LoRA adapter into its base model and save a standalone HF directory.

The output directory is what you point ``vllm serve`` at. The tokenizer
(including the chat template we set during training) is also written so
that vLLM renders prompts identically to how the trainer did.

Usage:
    python training/merge_lora.py \
        --adapter out/qwen25-05b-tutor-lora-v0 \
        --out     out/qwen25-05b-tutor-merged-v0
    # Optional override if the adapter does not record the base model id:
    python training/merge_lora.py --adapter out/... --out out/... \
        --base-model Qwen/Qwen2.5-0.5B-Instruct
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_LOG = logging.getLogger("training.merge")


def _resolve_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _resolve_base_model(adapter_dir: Path, override: str | None) -> str:
    if override:
        return override
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"adapter_config.json not found in {adapter_dir}; pass --base-model"
        )
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    base = cfg.get("base_model_name_or_path")
    if not base:
        raise ValueError(
            f"base_model_name_or_path missing from {cfg_path}; pass --base-model"
        )
    return base


def merge(adapter_dir: Path, out_dir: Path, base_model: str | None, dtype: str) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = _resolve_base_model(adapter_dir, base_model)
    _LOG.info("base model: %s", base)
    _LOG.info("adapter dir: %s", adapter_dir)
    _LOG.info("output dir: %s", out_dir)

    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype]

    out_dir.mkdir(parents=True, exist_ok=True)

    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch_dtype,
    )
    peft_model = PeftModel.from_pretrained(base_model_obj, str(adapter_dir))
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(str(out_dir), safe_serialization=True)

    # Prefer the tokenizer saved next to the adapter (it carries the chat
    # template we set during training); fall back to the base model's
    # tokenizer if the adapter dir doesn't have one.
    tokenizer_source = adapter_dir if (adapter_dir / "tokenizer_config.json").exists() else base
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source))
    tokenizer.save_pretrained(str(out_dir))

    _LOG.info("merged model + tokenizer written to %s", out_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--base-model",
        default=None,
        help="HF id of the base model. Defaults to value in adapter_config.json.",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    adapter_dir = _resolve_path(args.adapter)
    out_dir = _resolve_path(args.out)
    if not adapter_dir.exists():
        print(f"error: adapter dir not found: {adapter_dir}", file=sys.stderr)
        return 2
    if out_dir.exists() and any(out_dir.iterdir()):
        _LOG.warning("output dir %s is non-empty; files may be overwritten", out_dir)

    try:
        merge(adapter_dir, out_dir, args.base_model, args.dtype)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

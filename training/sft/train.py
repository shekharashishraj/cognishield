"""SFT entrypoint: `python -m training.sft.train --config <path> [overrides...]`."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from training.configs._schema import SFTConfig
from training.configs.loader import load_config
from training.data.cognibench_loader import build_sft_dataset


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pareto Tutors SFT warm-start.")
    ap.add_argument("--config", required=True, help="path to YAML")
    ap.add_argument(
        "--override",
        action="append",
        default=[],
        help="dot-path override, e.g. train.max_steps=10",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config, SFTConfig, overrides=args.override)
    print(f"[sft] config:\n{cfg.model_dump_json(indent=2)}", flush=True)

    # Defer heavy imports until after config validation so smoke fails fast.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ.setdefault("WANDB_MODE", cfg.wandb.mode)
    if cfg.wandb.enabled:
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb.project)
        if cfg.wandb.entity:
            os.environ.setdefault("WANDB_ENTITY", cfg.wandb.entity)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name,
        revision=cfg.model.revision,
        trust_remote_code=cfg.model.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        cfg.model.dtype
    ]
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        revision=cfg.model.revision,
        torch_dtype=dtype,
        trust_remote_code=cfg.model.trust_remote_code,
        attn_implementation=cfg.model.attn,
    )

    if cfg.peft.enabled:
        from peft import LoraConfig, get_peft_model

        lora_cfg = LoraConfig(
            r=cfg.peft.r,
            lora_alpha=cfg.peft.alpha,
            lora_dropout=cfg.peft.dropout,
            target_modules=cfg.peft.target_modules,
            task_type="CAUSAL_LM",
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    if cfg.train.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    train_ds, val_ds = build_sft_dataset(
        cfg.data.path,
        tokenizer=tokenizer,
        max_seq_len=cfg.data.max_seq_len,
        val_fraction=cfg.data.val_fraction,
        seed=cfg.data.seed,
        keep_splits=cfg.data.keep_splits,
    )
    print(
        f"[sft] dataset: train={len(train_ds)}, val={len(val_ds) if val_ds else 0}",
        flush=True,
    )

    # Late import so HF Trainer + accelerate can be missing in CPU-only envs.
    from training.sft.trainer import build_trainer

    trainer = build_trainer(cfg, model=model, tokenizer=tokenizer, train_ds=train_ds, val_ds=val_ds)
    trainer.train()
    trainer.save_model(str(Path(cfg.out_dir) / "final"))
    tokenizer.save_pretrained(str(Path(cfg.out_dir) / "final"))
    print(f"[sft] done. checkpoint at {Path(cfg.out_dir) / 'final'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

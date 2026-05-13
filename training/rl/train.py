"""GRPO entrypoint: `python -m training.rl.train --config <path> [overrides...]`."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List

from training.configs._schema import GRPOConfig
from training.configs.loader import load_config
from training.data.cognibench_loader import filter_accepted, iter_jsonl


def _build_prompt_dataset(cfg: GRPOConfig):
    """Each GRPO 'prompt' is a JSON spec blob — the rollout reads problem + ground_truth from it."""
    from datasets import Dataset

    rows: List[dict] = []
    for c in filter_accepted(iter_jsonl(cfg.data.path), keep_splits=cfg.data.keep_splits):
        if not c.problem or not c.solution:
            continue
        spec_blob = {
            "problem": c.problem,
            "ground_truth": c.solution,
            "subject": c.subject,
            "split": c.split,
        }
        rows.append({"prompt": json.dumps(spec_blob)})
    if not rows:
        raise ValueError(f"no usable problems found in {cfg.data.path}")
    return Dataset.from_list(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pareto Tutors GRPO training.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args(argv)

    cfg = load_config(args.config, GRPOConfig, overrides=args.override)
    print(f"[grpo] config:\n{cfg.model_dump_json(indent=2)}", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig as TRLGRPOConfig
    from trl import GRPOTrainer

    os.environ.setdefault("WANDB_MODE", cfg.wandb.mode)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.init_from if Path(cfg.init_from).exists() else cfg.model.name,
        trust_remote_code=cfg.model.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        cfg.model.dtype
    ]
    model = AutoModelForCausalLM.from_pretrained(
        cfg.init_from if Path(cfg.init_from).exists() else cfg.model.name,
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

    if cfg.model.attn == "flash_attention_2":
        # Flash-attn forbids gradient_checkpointing's use_reentrant=True default
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    from training.rl.grpo_trainer import make_reward_func
    from training.rl.tutor_backends import TransformersTutor, make_tutor_fn

    if cfg.infer.backend == "transformers":
        tutor_fn = TransformersTutor(
            model=model, tokenizer=tokenizer, temperature=cfg.rollout.temperature
        )
    else:
        tutor_fn = make_tutor_fn(cfg.infer.backend, model=model, tokenizer=tokenizer)

    reward_log = Path(cfg.out_dir) / "rollouts.jsonl"
    reward_fn = make_reward_func(cfg, tutor_fn=tutor_fn, log_jsonl=reward_log)

    train_ds = _build_prompt_dataset(cfg)

    trl_cfg = TRLGRPOConfig(
        output_dir=cfg.out_dir,
        per_device_train_batch_size=cfg.batch_problems,
        num_generations=cfg.group_size,
        gradient_accumulation_steps=1,
        learning_rate=cfg.lr,
        beta=cfg.kl_beta,
        num_iterations=cfg.mu_grad_steps,
        max_steps=cfg.total_updates,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=3,
        bf16=cfg.model.dtype == "bfloat16",
        seed=cfg.seed,
        report_to=["wandb"] if cfg.wandb.enabled else [],
        run_name=cfg.wandb.run_name,
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=trl_cfg,
        train_dataset=train_ds,
    )
    trainer.train()
    trainer.save_model(str(Path(cfg.out_dir) / "final"))
    tokenizer.save_pretrained(str(Path(cfg.out_dir) / "final"))
    print(f"[grpo] done. checkpoint at {Path(cfg.out_dir) / 'final'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

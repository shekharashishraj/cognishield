"""SFT trainer wiring — HF Trainer with the masked-loss collator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from training.configs._schema import SFTConfig
from training.sft.collator import MaskedCausalLMCollator


def build_trainer(cfg: SFTConfig, model: Any, tokenizer: Any, train_ds, val_ds: Optional[Any]):
    """Return a configured `transformers.Trainer`."""
    from transformers import Trainer, TrainingArguments

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eff_bs = cfg.train.eff_bs or (cfg.train.per_device_bs * cfg.train.grad_accum)
    _ = eff_bs  # documented; not directly consumed by HF Trainer

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg.train.epochs,
        max_steps=cfg.train.max_steps if cfg.train.max_steps else -1,
        per_device_train_batch_size=cfg.train.per_device_bs,
        per_device_eval_batch_size=cfg.train.per_device_bs,
        gradient_accumulation_steps=cfg.train.grad_accum,
        gradient_checkpointing=cfg.train.grad_ckpt,
        learning_rate=cfg.optim.lr,
        adam_beta1=cfg.optim.betas[0],
        adam_beta2=cfg.optim.betas[1],
        weight_decay=cfg.optim.weight_decay,
        lr_scheduler_type=cfg.optim.scheduler,
        warmup_ratio=cfg.optim.warmup_ratio,
        optim=cfg.optim.optimizer,
        logging_steps=cfg.train.logging_steps,
        save_steps=cfg.train.save_steps,
        eval_steps=cfg.train.eval_steps if val_ds is not None else None,
        eval_strategy="steps" if val_ds is not None else "no",
        save_strategy="steps",
        save_total_limit=3,
        bf16=cfg.model.dtype == "bfloat16",
        fp16=cfg.model.dtype == "float16",
        report_to=["wandb"] if cfg.wandb.enabled else [],
        run_name=cfg.wandb.run_name,
        seed=cfg.train.seed,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    collator = MaskedCausalLMCollator(tokenizer=tokenizer, max_length=cfg.data.max_seq_len)

    return Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

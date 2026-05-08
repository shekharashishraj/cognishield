"""SFT training entrypoint for the CogniShield tutor.

Loads ``training/data/sft.jsonl`` (produced by ``training/convert.py``)
and fine-tunes a small instruct model with TRL's :class:`SFTTrainer` and
PEFT LoRA. Honors per-turn loss masking via ``assistant_only_loss=True``,
which automatically sets ``label = -100`` on every non-assistant span as
long as every assistant turn was authored with ``loss_mask: true``
(enforced by ``training/convert.py``).

Device autodetect:
- CUDA -> bf16 LoRA, optional 4-bit QLoRA via ``--four-bit``.
- Apple MPS -> bf16 LoRA, no bitsandbytes (not supported on macOS).
- CPU -> fp32, prints a warning that this is slow.

Logs train loss after every step both to stdout and to
``<output_dir>/training_log.jsonl`` so ``smoke_eval.py`` can assert
"loss cratered" without parsing the trainer's pickled state.

Usage:
    python training/train_sft.py
    python training/train_sft.py --config training/configs/qwen2_5_0_5b.yaml
    python training/train_sft.py --four-bit             # CUDA only
    python training/train_sft.py --epochs 5 --lr 5e-5   # quick overrides
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "training" / "configs" / "qwen2_5_0_5b.yaml"

_LOG = logging.getLogger("training.sft")

# Qwen2.5 / ChatML template with `{% generation %}` markers so TRL's
# `assistant_only_loss=True` can mask everything except the assistant
# content + closing `<|im_end|>`. We override the model's default chat
# template because the stock Qwen2.5 template lacks generation markers
# and `assistant_only_loss` raises a RuntimeError without them. The
# template is saved with the tokenizer at the end of training, so vLLM
# serves the merged model with the same rendering.
QWEN_CHATML_GENERATION_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'user' %}"
    "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'assistant' %}"
    "<|im_start|>assistant\n"
    "{% generation %}{{ message['content'] }}<|im_end|>{% endgeneration %}\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


# ---------------------------------------------------------------------------
# Config loading.
# ---------------------------------------------------------------------------


@dataclass
class LoraSpec:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class TrainConfig:
    base_model: str
    train_file: Path
    output_dir: Path
    learning_rate: float = 1.0e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    seed: int = 42
    max_seq_length: int = 4096
    packing: bool = False
    logging_steps: int = 1
    save_strategy: str = "epoch"
    report_to: list[str] = field(default_factory=list)
    chat_template: str | None = None
    trust_remote_code: bool = False
    lora: LoraSpec = field(default_factory=LoraSpec)


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml  # local import so --help doesn't require the dep

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (REPO_ROOT / p)


def load_config(path: Path) -> TrainConfig:
    raw = _load_yaml(path) or {}
    lora_raw = raw.pop("lora", {}) or {}
    lora = LoraSpec(
        r=int(lora_raw.get("r", 16)),
        alpha=int(lora_raw.get("alpha", 32)),
        dropout=float(lora_raw.get("dropout", 0.05)),
        target_modules=list(
            lora_raw.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
        ),
    )
    return TrainConfig(
        base_model=raw["base_model"],
        train_file=_resolve_path(raw["train_file"]),
        output_dir=_resolve_path(raw.get("output_dir", "out/sft")),
        learning_rate=float(raw.get("learning_rate", 1.0e-4)),
        num_train_epochs=int(raw.get("num_train_epochs", 3)),
        per_device_train_batch_size=int(raw.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(raw.get("gradient_accumulation_steps", 4)),
        warmup_ratio=float(raw.get("warmup_ratio", 0.05)),
        weight_decay=float(raw.get("weight_decay", 0.0)),
        lr_scheduler_type=str(raw.get("lr_scheduler_type", "cosine")),
        max_grad_norm=float(raw.get("max_grad_norm", 1.0)),
        seed=int(raw.get("seed", 42)),
        max_seq_length=int(raw.get("max_seq_length", 4096)),
        packing=bool(raw.get("packing", False)),
        logging_steps=int(raw.get("logging_steps", 1)),
        save_strategy=str(raw.get("save_strategy", "epoch")),
        report_to=list(raw.get("report_to", []) or []),
        chat_template=raw.get("chat_template"),
        trust_remote_code=bool(raw.get("trust_remote_code", False)),
        lora=lora,
    )


# ---------------------------------------------------------------------------
# Device / dtype detection.
# ---------------------------------------------------------------------------


@dataclass
class DeviceProfile:
    device: str           # "cuda" / "mps" / "cpu"
    use_bf16: bool
    use_fp16: bool
    use_4bit: bool


def detect_device(four_bit_requested: bool) -> DeviceProfile:
    import torch

    if torch.cuda.is_available():
        if four_bit_requested:
            try:
                import bitsandbytes  # noqa: F401
            except ImportError:
                _LOG.warning(
                    "--four-bit requested but bitsandbytes is not installed; "
                    "falling back to bf16 LoRA"
                )
                four_bit_requested = False
        return DeviceProfile(device="cuda", use_bf16=True, use_fp16=False, use_4bit=four_bit_requested)

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        if four_bit_requested:
            _LOG.warning(
                "--four-bit is unavailable on Apple MPS (bitsandbytes is "
                "Linux/CUDA-only); using bf16 LoRA instead"
            )
        return DeviceProfile(device="mps", use_bf16=True, use_fp16=False, use_4bit=False)

    _LOG.warning(
        "No CUDA or MPS device detected; falling back to CPU. Training a "
        "0.5B model on CPU will be very slow even at n=10."
    )
    if four_bit_requested:
        _LOG.warning("--four-bit ignored on CPU.")
    return DeviceProfile(device="cpu", use_bf16=False, use_fp16=False, use_4bit=False)


# ---------------------------------------------------------------------------
# Training-log callback (writes JSONL alongside the adapter).
# ---------------------------------------------------------------------------


def _build_log_callback(log_path: Path):
    from transformers.trainer_callback import TrainerCallback

    class JsonlLogger(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            record = {"step": state.global_step, "epoch": state.epoch, **logs}
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return JsonlLogger()


# ---------------------------------------------------------------------------
# Main training routine.
# ---------------------------------------------------------------------------


def _override_from_args(cfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    if args.epochs is not None:
        cfg.num_train_epochs = args.epochs
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.output_dir is not None:
        cfg.output_dir = _resolve_path(args.output_dir)
    if args.train_file is not None:
        cfg.train_file = _resolve_path(args.train_file)
    return cfg


def train(cfg: TrainConfig, four_bit: bool, dry_run: bool) -> None:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(cfg.seed)

    profile = detect_device(four_bit_requested=four_bit)
    _LOG.info(
        "device=%s bf16=%s fp16=%s 4bit=%s",
        profile.device,
        profile.use_bf16,
        profile.use_fp16,
        profile.use_4bit,
    )

    if not cfg.train_file.exists():
        raise FileNotFoundError(
            f"Training file not found: {cfg.train_file}. "
            "Run `python training/convert.py` first."
        )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.output_dir / "training_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model, trust_remote_code=cfg.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # `assistant_only_loss=True` requires `{% generation %}` markers in the
    # chat template. Inject our Qwen-compatible template if the model's
    # default lacks them; explicit `chat_template` config wins over both.
    if cfg.chat_template:
        tokenizer.chat_template = cfg.chat_template
    elif "{% generation %}" not in (tokenizer.chat_template or ""):
        _LOG.info(
            "tokenizer chat_template lacks {%% generation %%} markers; "
            "overriding with Qwen-style ChatML template"
        )
        tokenizer.chat_template = QWEN_CHATML_GENERATION_TEMPLATE

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": cfg.trust_remote_code,
    }
    if profile.use_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["device_map"] = "auto"
    else:
        if profile.use_bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif profile.use_fp16:
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **model_kwargs)
    if profile.device == "mps":
        model.to("mps")

    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = load_dataset("json", data_files=str(cfg.train_file), split="train")
    _LOG.info("loaded %s train records from %s", len(dataset), cfg.train_file)

    sft_args = SFTConfig(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        seed=cfg.seed,
        bf16=profile.use_bf16,
        fp16=profile.use_fp16,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        report_to=cfg.report_to,
        max_seq_length=cfg.max_seq_length,
        packing=cfg.packing,
        assistant_only_loss=True,
        gradient_checkpointing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[_build_log_callback(log_path)],
    )

    if dry_run:
        _LOG.info("--dry-run enabled; skipping trainer.train()")
        return

    trainer.train()
    trainer.save_model(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))
    _LOG.info("saved adapter + tokenizer to %s", cfg.output_dir)
    _LOG.info("training log written to %s", log_path)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--four-bit", action="store_true", help="QLoRA (CUDA only)")
    parser.add_argument("--dry-run", action="store_true", help="Build trainer but skip train()")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    if not args.config.exists():
        print(f"error: config not found: {args.config}", file=sys.stderr)
        return 2

    cfg = _override_from_args(load_config(args.config), args)
    train(cfg, four_bit=args.four_bit, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

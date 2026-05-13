"""Config loader composes YAML + CLI overrides → validated pydantic object."""

from __future__ import annotations

from pathlib import Path

from training.configs._schema import GRPOConfig, SFTConfig
from training.configs.loader import load_config


REPO = Path(__file__).resolve().parents[2]


def test_load_sft_smoke_config() -> None:
    cfg = load_config(REPO / "training/configs/sft/qwen35_9b_smoke.yaml", SFTConfig)
    assert cfg.model.name == "Qwen/Qwen3.5-9B"
    assert cfg.train.max_steps == 10
    assert cfg.mask_student_turns is True


def test_load_sft_full_config_default_full_ft() -> None:
    cfg = load_config(REPO / "training/configs/sft/qwen35_9b_full.yaml", SFTConfig)
    assert cfg.train.full_finetune is True
    assert cfg.peft.enabled is False


def test_load_grpo_smoke_uses_mocks() -> None:
    cfg = load_config(REPO / "training/configs/rl/grpo_smoke.yaml", GRPOConfig)
    assert cfg.judges.mock is True
    assert cfg.student.mock is True
    assert cfg.total_updates == 3


def test_overrides_apply() -> None:
    cfg = load_config(
        REPO / "training/configs/rl/grpo_full.yaml",
        GRPOConfig,
        overrides=["lr=1e-6", "rewards.lambda_safety=0.0", "total_updates=10"],
    )
    assert cfg.lr == 1e-6
    assert cfg.rewards.lambda_safety == 0.0
    assert cfg.total_updates == 10

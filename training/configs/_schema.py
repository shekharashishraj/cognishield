"""Pydantic schemas for training YAML configs.

Configs are written as YAML and loaded with `omegaconf` for composition + CLI
overrides, then validated by these models before any trainer starts.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "Qwen/Qwen3.5-9B"
    revision: Optional[str] = None
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    attn: Literal["flash_attention_2", "sdpa", "eager"] = "flash_attention_2"
    trust_remote_code: bool = True


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    max_seq_len: int = 4096
    val_fraction: float = 0.05
    seed: int = 42
    keep_splits: List[str] = Field(
        default_factory=lambda: ["exemplary_legitimate", "adequate_ambiguous"]
    )


class OptimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lr: float = 2e-5
    betas: List[float] = Field(default_factory=lambda: [0.9, 0.95])
    weight_decay: float = 0.0
    scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    warmup_ratio: float = 0.03
    optimizer: Literal["adamw_torch", "adamw_torch_fused", "paged_adamw_32bit"] = (
        "adamw_torch_fused"
    )


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: int = 1
    max_steps: Optional[int] = None
    per_device_bs: int = 4
    grad_accum: int = 8
    eff_bs: Optional[int] = None
    grad_ckpt: bool = True
    full_finetune: bool = True
    save_steps: int = 200
    eval_steps: int = 100
    logging_steps: int = 5
    seed: int = 42


class PeftConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target_modules: List[str] = Field(
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


class WandbConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    project: str = "pareto-tutors"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    mode: Literal["online", "offline", "disabled"] = "offline"


class SFTConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig
    optim: OptimConfig = Field(default_factory=OptimConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    peft: PeftConfig = Field(default_factory=PeftConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    mask_student_turns: bool = True
    out_dir: str = "checkpoints/tutor_sft"


class RewardsWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lambda_ped: float = 1.0
    lambda_safety: float = 1.0
    lambda_age: float = 1.0
    aux_template: float = 0.1
    aux_eoc: float = 0.2
    aux_length: float = 0.1
    length_budget_tokens_per_turn: int = 512


class JudgesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoint: str = "http://localhost:8001/v1"
    model: str = "Qwen/Qwen2.5-14B-Instruct-AWQ"
    api_key: str = "EMPTY"
    temperature: float = 0.0
    max_tokens: int = 512
    timeout: float = 30.0
    mock: bool = False


class StudentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoint: str = "http://localhost:8002/v1"
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    api_key: str = "EMPTY"
    temperature: float = 1.0
    max_tokens: int = 512
    timeout: float = 30.0
    mock: bool = False
    solve_samples_k: int = 8


class InferConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["vllm", "transformers"] = "vllm"
    gpu_memory_utilization: float = 0.45
    max_model_len: int = 4096


class RolloutConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_dialog_turns: int = 20
    min_dialog_turns: int = 6
    end_token: str = "<end_of_conversation>"
    temperature: float = 1.0


class GRPOConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelConfig = Field(default_factory=ModelConfig)
    init_from: str = "checkpoints/tutor_sft"
    data: DataConfig
    peft: PeftConfig = Field(
        default_factory=lambda: PeftConfig(enabled=True)
    )

    group_size: int = 8
    batch_problems: int = 32
    mu_grad_steps: int = 2
    lr: float = 5e-7
    kl_beta: float = 0.005
    total_updates: int = 400
    save_steps: int = 50
    logging_steps: int = 1
    seed: int = 42

    rewards: RewardsWeights = Field(default_factory=RewardsWeights)
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    infer: InferConfig = Field(default_factory=InferConfig)
    judges: JudgesConfig = Field(default_factory=JudgesConfig)
    student: StudentConfig = Field(default_factory=StudentConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    out_dir: str = "checkpoints/tutor_grpo"

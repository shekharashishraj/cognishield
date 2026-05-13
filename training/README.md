# Training pipelines (SFT + GRPO) — Pareto Tutors on Qwen3.5-9B

Implements the Pareto Tutors paper's `SFT → GRPO → (optional) RFT` recipe on
Qwen/Qwen3.5-9B. Modular by design: every phase has a YAML config, every
reward axis is a separate file, judge prompts are plain text, smoke tests
exercise the whole pipeline before any real training run.

## Layout

```
training/
├── configs/          YAML, validated by training/configs/_schema.py
├── data/             cognibench loader, chat template, tutor-turn masking
├── sft/              Phase 1 — masked-loss SFT (full FT)
├── rl/               Phase 2 — multi-objective GRPO (LoRA)
├── rewards/          r_sol, r_ped, r_safety, r_age, aux, aggregator
├── judges/           vLLM client + 3 judge axes + prompts
├── verifiers/        math (numeric exact-match); code is v2
├── rft/              Phase 3 — rejection-sampling refresh
└── scripts/          start_*_server.sh, smoke_*.sh, run_full_*.sh
```

## Install

Training extras (on the H200 host):

```bash
pip install -r requirements.txt -r requirements-train.txt
# GPU wheels, install separately:
pip install vllm>=0.7 flash-attn>=2.6 bitsandbytes>=0.44
```

CPU dev only needs `pydantic`, `omegaconf`, `torch` (any version) to run
the test suite.

## Three phases

## Data

- **SFT data**: `sft.generated.batch.jsonl` (586 conversations across 3
  splits — `exemplary_legitimate`, `adequate_ambiguous`,
  `failing_disallowed` — all containing correct tutor behavior). Format is
  OpenAI chat: `{conversation_id, split, messages:[{role, content}, ...]}`
  with `role ∈ {system, user, assistant}`. The loader keeps all three
  splits by default.
- **GRPO data**: a *separate* problems file with `problem` + `solution`
  fields per record (math GT for `r_sol`). Not provided by the SFT file —
  point `data.path` at e.g. a BigMath-filtered JSONL.

### Phase 1 — SFT warm-start

Full fine-tune on the SFT JSONL with **assistant-turn-only loss**
(system + user tokens are masked to `-100`). Paper §5.1: LR 2e-5, batch
32, 1 epoch.

```bash
# smoke (~5 min on H200)
bash training/scripts/smoke_sft.sh

# full run
bash training/scripts/run_full_sft.sh
# or: accelerate launch -m training.sft.train --config training/configs/sft/qwen35_9b_full.yaml
```

### Phase 2 — Multi-objective GRPO

Initializes from `checkpoints/tutor_sft/final/`. **LoRA** on the tutor for
single-H200 co-residency with the judge and student vLLM servers. Reward:

```
r = r_sol + λ_ped·(r_ped−1)
        + λ_safety·(r_safety−1)·1[c>0]
        + λ_age   ·(r_age   −1)·1[α≠adult]
        + r_aux
```

Each of 8 group rollouts uses a different `(persona, age_band, coercion)`
spec sampled from `training/rl/spec_sampler.py`. Pareto-weight sweeps are a
single override:

```bash
python -m training.rl.train --config training/configs/rl/grpo_full.yaml \
    --override rewards.lambda_safety=0.5 \
    --override rewards.lambda_age=2.0
```

Before launching the full run, **start the helper servers**:

```bash
bash training/scripts/start_judge_server.sh   &   # :8001
bash training/scripts/start_student_server.sh &   # :8002
bash training/scripts/run_full_grpo.sh             # ~36-48 h on H200
```

Smoke test (mocked judge + student, transformers tutor backend, ~15 min):

```bash
bash training/scripts/smoke_grpo.sh
```

### Phase 3 (optional) — Rejection-Sampling Refresh

```bash
python -m training.rft.train \
    --rollouts checkpoints/tutor_grpo/rollouts.jsonl \
    --out_data rollouts/top_quartile.jsonl \
    --sft_config training/configs/sft/qwen35_9b_full.yaml \
    --override optim.lr=1e-5 out_dir=checkpoints/tutor_rft
```

## Config system

YAML files are loaded with omegaconf (composition + dot-path overrides) and
validated by pydantic models in `training/configs/_schema.py`. Any field on
any model is overridable on the CLI:

```bash
python -m training.sft.train \
    --config training/configs/sft/qwen35_9b_full.yaml \
    --override data.path=cognibench.jsonl \
    --override train.max_steps=100 \
    --override wandb.enabled=true
```

## Reward axes — graceful degradation

The paper assumes the data pipeline emits `age_band`, `student_persona`,
`kora_*`. The current `cognibench.jsonl` doesn't — so:

- `r_sol` always runs (math GT comes from the existing `solution` field).
- `r_ped` always runs.
- `r_safety` runs only when `coercion_level > 0`; otherwise the gate is
  off and the axis contributes 0 to the total.
- `r_age` runs only when the spec sampler picks a non-adult age band;
  otherwise 0 contribution.

When the data pipeline gets richer, switch to `judge_accepted: true` rows
and replace `SpecSampler.sample()` with reading the spec straight off each
record — no other code change required.

## Memory budget on a single H200

| Component                          | VRAM     |
| ---------------------------------- | -------- |
| SFT, Qwen3.5-9B BF16 full FT       | ~120 GB  |
| GRPO, LoRA tutor (base + adapter)  | ~19 GB   |
| GRPO, vLLM KV @ 0.45 util          | ~45 GB   |
| Judge (Qwen2.5-14B-AWQ)            | ~7 GB    |
| Student (Llama-3.1-8B BF16)        | ~16 GB   |
| **GRPO total resident**            | **~87 GB / 141 GB** |

If actual usage exceeds this, lower `infer.gpu_memory_utilization`
and/or the judge `max_model_len` first; flip optimizer to
`paged_adamw_32bit` last.

## Tests

```bash
PYTHONPATH=. pytest tests/training/ -q
```

40 tests cover: reward gates, Pareto math, math verifier, spec-sampler
distribution, JSON parsing, judge clients on the mock backend, env pipeline
(end-to-end mock), config loader + override semantics, multi-turn rollout
driver, and tutor-turn masking on a fake fast tokenizer. No GPU needed.

## What's not here (v2)

- Code-domain `r_sol` (`training/verifiers/code.py` raises NotImplementedError).
- vLLM tutor backend — wired up but raises until `vllm>=0.7` ships Qwen3.5
  kernel support. Use `infer.backend=transformers` until then.
- Evaluation harness (MathTutorBench / EduBench / EduGuardBench / KORA).

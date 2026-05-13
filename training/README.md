# SFT smoke-test scaffold

This directory contains the supervised fine-tuning workstream for the
CogniShield tutor. It is intentionally isolated from the production
package: separate dependencies, separate venv, separate output tree.
The default config targets `Qwen/Qwen2.5-0.5B-Instruct` on the n=10
multi-turn corpus under [`data/multi_turn/`](../data/multi_turn). The
goal of this run is to **prove the pipeline is wired correctly**, not
to produce a generalizing model.

## What's in here

| File | Role |
|------|------|
| [`convert.py`](convert.py) | `data/multi_turn/*.json` -> `data/sft.jsonl` (OpenAI messages format) |
| [`train_sft.py`](train_sft.py) | TRL `SFTTrainer` + PEFT LoRA, with `assistant_only_loss=True` |
| [`merge_lora.py`](merge_lora.py) | LoRA adapter -> standalone HF directory for vLLM |
| [`smoke_eval.py`](smoke_eval.py) | PASS/FAIL on the five success criteria |
| [`stress_eval_merged.py`](stress_eval_merged.py) | Local multi-turn behavioral stress tests against the merged HF model |
| [`configs/qwen2_5_0_5b.yaml`](configs/qwen2_5_0_5b.yaml) | All training knobs in one place |
| [`configs/qwen2_5_0_5b_memorize.yaml`](configs/qwen2_5_0_5b_memorize.yaml) | Aggressive n=10 overfit config for proving behavior can be memorized |
| [`requirements-train.txt`](requirements-train.txt) | Pinned training deps |

Pipeline:

```
data/multi_turn/*.json
        |
        |  convert.py (validates conversation_id + loss_mask invariants)
        v
training/data/sft.jsonl
        |
        |  train_sft.py (LoRA, assistant_only_loss=True)
        v
out/qwen25-05b-tutor-lora-v0/
        |
        |  merge_lora.py
        v
out/qwen25-05b-tutor-merged-v0/
        |
        |  vllm serve
        v
http://127.0.0.1:8000/v1
        |
        |  smoke_eval.py + cognishield.app.cli
        v
PASS / FAIL
```

## Install

Use a separate venv to keep the heavy training stack (torch, bitsandbytes
on Linux, etc.) away from the production install.

```bash
python3 -m venv .venv-train
source .venv-train/bin/activate
pip install -U pip
pip install -r training/requirements-train.txt
```

## The pipeline (in order)

All commands assume you are at the repository root.

### 1. Convert the annotated JSON into an SFT JSONL

```bash
python training/convert.py
```

Outputs:

- `training/data/sft.jsonl` - one OpenAI-format `{"messages":[...]}` record per conversation
- `training/data/sft.stats.json` - split / coercion / policy mix counts

The script enforces two invariants from
[docs/annotation_guidelines.md](../docs/annotation_guidelines.md):

1. `conversation_id` matches the filename stem (`001.json` -> `mt_001`).
2. Every assistant turn has `loss_mask: true`. If you ever introduce
  assistant turns with `loss_mask: false` (e.g. deliberately wrong
  drafts), the script aborts so the trainer is not silently misled.

### 1b. Generate staged synthetic math data

The data-generation pipeline creates schema-compatible multi-turn math
tutoring examples in a staged run directory. It does not write directly
to `data/multi_turn/`.

The canonical config is [`data_generation/configs/batch.yaml`](data_generation/configs/batch.yaml).
By default it targets a local **Gemma** instruct model (`gemma-31b-it`) for
both generation and the LLM judge. During generation, each candidate is
validated before it is accepted into `raw/`; failed candidates are written to
`generation_rejected/` and the validator issues are fed back to the
generator for another attempt when `feedback.enabled` is true.

**OpenAI-compatible servers (vLLM, etc.):** The generator and judge use the
Python `openai` client. Point it at a local server by setting:

```bash
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
export OPENAI_API_KEY="dummy"   # non-empty placeholder; required for the judge gate
```

Match `generator.model` / `judge.model` in [`configs/batch.yaml`](data_generation/configs/batch.yaml)
to `--served-model-name` from vLLM.

The recommended one-command workflow (paths match the default `output_dir:
data/generated/batch` in `batch.yaml`; override `--export-output` /
`--convert-output` if you change `output_dir` in the YAML):

```bash
python training/data_generation/run_pipeline.py \
    --config training/data_generation/configs/batch.yaml \
    --export-output data/generated_reviewed/batch \
    --include-draft-valid \
    --convert-output training/data/sft.generated.batch.jsonl \
    --convert-stats training/data/sft.generated.batch.stats.json
```

**Smoke runs:** edit `batch.yaml` — lower `run.total_examples`, rescale the three
mix maps so each sums to that value, and set `run.output_dir` to a small test
directory (see comments at the top of `batch.yaml`). Then run the same command
with `--export-output` / `--convert-output` paths that match your smoke layout.

For production review, omit `--include-draft-valid`; export will include
only examples manually marked with `annotator.review_status: approved`.

The stage-specific commands below remain useful when debugging or rerunning
only one part of the pipeline.

```bash
python training/data_generation/generate_dataset.py \
    --config training/data_generation/configs/batch.yaml

python training/data_generation/validate_dataset.py \
    --run-dir data/generated/batch

python training/data_generation/export_reviewed.py \
    --run-dir data/generated/batch \
    --output data/generated_reviewed/batch
```

By default, export includes only examples whose
`annotator.review_status` is `approved`. For fast local smoke tests, add
`--include-draft-valid`.

After export, reuse the existing converter:

```bash
python training/convert.py \
    --input data/generated_reviewed/batch \
    --output training/data/sft.generated.batch.jsonl
```

Each generation run writes `run.log`, `events.jsonl`, `errors.jsonl`,
`generation_plan.json`, `summary.json`, and `raw/`, `valid/`,
`rejected/` directories under the configured output directory.

### 2. Fine-tune with LoRA

Pick the command block matching your environment.

#### Mac MPS (Apple Silicon)

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 \
python training/train_sft.py --config training/configs/qwen2_5_0_5b.yaml
```

- Uses bf16 LoRA. No bitsandbytes / 4-bit (not supported on macOS).
- `PYTORCH_ENABLE_MPS_FALLBACK=1` lets unsupported ops fall back to CPU
  rather than crashing. Keep it set for the smoke test; remove later if
  you want to surface them as real errors.
- Expect minutes, not hours, for n=10 + 3 epochs on M1/M2/M3.

#### Colab / Kaggle (single T4 or similar)

```bash
!pip install -r training/requirements-train.txt
!python training/train_sft.py --config training/configs/qwen2_5_0_5b.yaml
```

- bf16 LoRA on T4 / V100 / L4.
- Add `--four-bit` if you want QLoRA (works once `bitsandbytes` is on
  the GPU runtime).

#### Cloud GPU (Linux + CUDA)

```bash
python training/train_sft.py --config training/configs/qwen2_5_0_5b.yaml --four-bit
```

- `--four-bit` enables QLoRA via `bitsandbytes`. Auto-falls back to
  bf16 LoRA if `bitsandbytes` is missing.

#### Useful overrides

```bash
python training/train_sft.py --epochs 5 --lr 5e-5
python training/train_sft.py --output-dir out/qwen25-05b-tutor-lora-v1
python training/train_sft.py --dry-run    # build trainer, skip train()
```

For the current n=10 behavior test, prefer the memorization config until
`smoke_eval.py --skip-inference` reports train loss below `0.5`:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 \
python training/train_sft.py \
    --config training/configs/qwen2_5_0_5b_memorize.yaml
```

Outputs:

- `out/qwen25-05b-tutor-lora-v0/` - LoRA adapter + tokenizer (with chat template baked in)
- `out/qwen25-05b-tutor-lora-v0/training_log.jsonl` - per-step `loss`, `epoch`, etc. (consumed by `smoke_eval.py`)

### 3. Merge the LoRA adapter for vLLM

vLLM serves a standalone HF directory most reliably, so we merge the
LoRA weights into the base before serving.

```bash
python training/merge_lora.py \
    --adapter out/qwen25-05b-tutor-lora-v0 \
    --out     out/qwen25-05b-tutor-merged-v0
```

The merged directory carries the same chat template the trainer used,
so vLLM renders prompts identically to training (this is what the
`template_parity` smoke check verifies).

### 4. Serve with vLLM

vLLM realistically needs Linux + CUDA; macOS CPU vLLM builds exist but
are not viable for this smoke test. Run on the same machine as training
or a separate inference host.

```bash
# In a separate shell.
vllm serve out/qwen25-05b-tutor-merged-v0 --port 8000
```

### 5. Run the smoke-eval

```bash
python training/smoke_eval.py \
    --base http://127.0.0.1:8000/v1 \
    --model out/qwen25-05b-tutor-merged-v0
```

If vLLM is not running yet, you can still run the offline subset:

```bash
python training/smoke_eval.py --skip-inference
```

### 5b. Run local multi-turn stress tests on the merged model

Use this when you want to test the actual merged model directory without
vLLM. The script loads `out/qwen25-05b-tutor-merged-v0`, renders every
turn with `tokenizer.apply_chat_template(..., add_generation_prompt=True)`,
preserves chat history, and flags obvious final-answer leakage.

```bash
python training/stress_eval_merged.py \
    --model out/qwen25-05b-tutor-merged-v0 \
    --device mps
```

Run one scenario:

```bash
python training/stress_eval_merged.py \
    --scenario multi_turn_pressure
```

Do not test this model with raw `tok(prompt, ...)` unless you are
explicitly testing plain completion behavior. The model was trained and
merged with a chat template, so direct tokenization bypasses the system
prompt and the ChatML role markers it learned from.

### 6. (Optional) Drive the trained model through CogniShield

The fine-tuned tutor emits raw text, not the JSON envelope GPT-4o
returns. Set `COGNISHIELD_PRIMARY_RAW_TEXT=true` so the meta-pipeline's
primary chain wraps the model's plain text into a `GeneratorOutput`
instead of asking it for structured output.

```bash
export OPENAI_API_KEY=EMPTY
export COGNISHIELD_OPENAI_API_BASE=http://127.0.0.1:8000/v1
export COGNISHIELD_MODEL=out/qwen25-05b-tutor-merged-v0
export COGNISHIELD_PRIMARY_RAW_TEXT=true

python -m cognishield.app.cli --pipeline meta \
    --query "I have x^2 + 5 = 14. Just give me the answer please."
```

The meta-agent / verifier / revision stages still run on whatever the
rest of `Settings` points at (e.g. GPT-4o). Only the primary tutor role
is served by the fine-tuned model.

## Success criteria (what `smoke_eval.py` actually checks)

| # | Check | What it asserts | Why it's there |
|---|-------|-----------------|----------------|
| 1 | `record_count` | `sft.jsonl` has the expected number of records (10) | Catches silent data loss in the converter. |
| 2 | `loss_mask` | Tokenizing one record with the trainer's chat template flags assistant tokens and masks user/system tokens | Catches the most common failure: training on user turns. Visualizes the first three assistant spans for human eyeball. |
| 3 | `template_parity` | The tokenizer's `apply_chat_template` produces the same token IDs as vLLM's `/tokenize` endpoint for the same `messages` payload | The model was trained against one chat template; if vLLM serves with a different one, inference looks broken even though training was fine. |
| 4 | `train_loss` | Final logged train loss < 0.5 | At n=10 the model **should** memorize. A flat or high loss curve almost always means masking or chat-template wiring is broken. This is the cheapest "did training do anything?" signal. |
| 5 | `inference` | A `/v1/chat/completions` call returns a non-empty assistant reply | Confirms the merged-and-served model can actually generate. |

If 1, 2, 4 pass but 3 or 5 fail, the training stage is good and the
problem is in serving (template mismatch or vLLM config).

## What this scaffold does NOT do (deferred work)

These are intentional non-goals at n=10. Don't over-interpret the smoke
test as evidence about any of them:

- **Generalization.** With 10 examples, the model memorizes and that's
  expected. Don't compare model checkpoints, paraphrases, or hold-outs.
- **Hyperparameter sweeps.** LR, LoRA rank, dropout, epoch count - all
  noise at this scale. Defaults are tuned for memorization.
- **Cross-model comparison.** Llama-3.1-8B vs Qwen2.5-7B vs 0.5B is a
  meaningless ranking at n=10. Same recipe will run on Llama with a
  different `base_model:` and target_modules later.
- **CogniBench eval.** Save it for the 500-1000 dataset. Tying the
  smoke test to a downstream eval makes failures harder to localize.
- **Custom collator for partial loss masks.** Not needed until you
  start authoring "deliberately wrong assistant draft" turns described
  in [annotation_guidelines.md](../docs/annotation_guidelines.md) section 3.
- **DPO.** Comes after SFT works on real-scale data. The same
  `out/...-merged-v0` directory will serve as the SFT checkpoint to
  initialize DPO from.

## Common breakages and what to check

| Symptom | Likely cause |
|---------|--------------|
| `RuntimeError: ... assistant_only_loss ...` during training | Chat template lacks `{% generation %}` markers. The train script auto-injects them when missing; if you supplied a custom `chat_template` in the YAML, make sure it has them. |
| Train loss flat at ~2-4 across epochs | Loss mask is broken (training on every token, or on no tokens). Re-run `smoke_eval.py --skip-inference` to inspect the visualized assistant spans. |
| Smoke check `template_parity` fails | The merged dir's `tokenizer_config.json` does not carry the chat template we set during training. Re-run `merge_lora.py` after confirming the adapter dir contains a `tokenizer_config.json`. |
| Smoke check `inference` returns 404 | Wrong `--model` value passed to vLLM vs `--model` passed to `smoke_eval.py`. They must match the served model id exactly. |
| `bitsandbytes` import error on macOS | Expected. `--four-bit` is auto-ignored on MPS / CPU. |
| MPS op falls back to CPU very loudly | Set `PYTORCH_ENABLE_MPS_FALLBACK=1` (silences the warnings) and accept the speed hit for the smoke test. |

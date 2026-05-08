# CogniShield

CogniShield is a prototype tutoring-safety system for educational AI. It
wraps an LLM tutor with planning, validation, revision, and tracing logic so
the assistant can help students learn without giving away disallowed final
answers or producing poor training behavior.

The repository has three related workstreams:

1. **Runtime shielding app** - a LangChain CLI that runs the CogniShield tutor
   pipeline against OpenAI or an OpenAI-compatible backend such as vLLM.
2. **Synthetic SFT data generation** - a staged pipeline that creates validated
   multi-turn math tutoring conversations for supervised fine-tuning.
3. **Training smoke tests** - scripts for converting the data, fine-tuning a
   small instruct model with LoRA, merging the adapter, and evaluating behavior.

## Repository Map

| Path | Purpose |
| --- | --- |
| `cognishield/app/` | Runtime package: CLI, settings, schemas, chains, prompts, orchestrator, verifier, tracing |
| `training/data_generation/` | Synthetic math conversation generation, validation, export, and conversion orchestration |
| `training/` | SFT conversion, LoRA training, merge, smoke-eval, and stress-eval scripts |
| `data/multi_turn/` | Small hand-authored multi-turn seed/eval examples |
| `data/generated/` | Staged generated data output; created by the data-generation pipeline |
| `data/generated_reviewed/` | Exported generated examples ready for SFT conversion |
| `docs/` | Architecture, vLLM inference, module reference, and annotation guidance |
| `tests/` | Unit and smoke tests |

## Requirements

- Python **3.10+**
- OpenAI API key for OpenAI-backed inference or data generation
- Optional: an OpenAI-compatible vLLM server for local/open-weight inference
- Optional for training: a separate Python environment with the heavier
  training stack from `training/requirements-train.txt`

## Quick Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

Set an OpenAI key for runtime inference:

```bash
export OPENAI_API_KEY="sk-..."
```

You can also create a `.env` file in the repo root:

```bash
OPENAI_API_KEY=sk-...
```

CogniShield-specific environment variables use the `COGNISHIELD_` prefix. For
example:

```bash
export COGNISHIELD_MODEL="gpt-4o"
export COGNISHIELD_PIPELINE="meta"
```

## Run CogniShield

Run one tutoring turn:

```bash
python3 -m cognishield.app.cli \
  --query "Can you give me a hint for this algebra homework without solving it?"
```

The final assistant text is printed on stdout. Logs are printed on stderr.

Useful variants:

```bash
# No API calls; verifies wiring and CLI parsing.
python3 -m cognishield.app.cli --query "test" --dry-run

# Use the legacy planner/generator/validator pipeline.
python3 -m cognishield.app.cli \
  --pipeline legacy \
  --query "Help me understand this step."

# Use the meta pipeline: primary -> meta-agent -> verifier -> revision.
python3 -m cognishield.app.cli \
  --pipeline meta \
  --query "Give me a hint only."

# Write structured JSONL traces.
python3 -m cognishield.app.cli \
  --query "..." \
  --trace-path ./runs/trace.jsonl
```

To discover all flags:

```bash
python3 -m cognishield.app.cli --help
```

## Runtime Pipelines

CogniShield supports two runtime orchestration modes.

| Pipeline | Flow | When to use |
| --- | --- | --- |
| `legacy` | planner -> generator -> Bloom/cognitive/safety/accuracy validators -> rule verifier -> optional revision loop | Original multi-validator prototype |
| `meta` | primary tutor -> meta-agent classifiers -> rule verifier -> revision | Simpler classifier-and-revision path |

Both modes consume a `TurnContext` with the user query, optional history,
learner profile, rubric constraints, and task context. See
[`docs/architecture.md`](docs/architecture.md) for diagrams and implementation
details.

## OpenAI-Compatible Inference With vLLM

If you have a vLLM server exposing the OpenAI-compatible API:

```bash
export OPENAI_API_KEY="EMPTY"
export COGNISHIELD_OPENAI_API_BASE="http://127.0.0.1:8000/v1"
export COGNISHIELD_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

python3 -m cognishield.app.cli \
  --query "Give me a scaffolded hint for this problem."
```

The model name must match the served vLLM model id. Full setup instructions are
in [`docs/vllm-inference.md`](docs/vllm-inference.md).

## Synthetic SFT Data Generation

The data-generation pipeline creates multi-turn math tutoring examples and
filters out bad examples before they become training data.

Current default config:

| Setting | Value |
| --- | --- |
| Config | `training/data_generation/configs/batch_001.yaml` |
| Target final examples | `100` |
| Candidate safety cap | `300` |
| Generator model | `gpt-4o` |
| LLM judge model | `gpt-5.1` |
| Regeneration attempts | `3` retries per candidate, so up to 4 total attempts |
| Final validation | Deterministic local validation only; the LLM judge already ran during generation |

Important behavior:

- `total_examples: 100` means the pipeline tries to produce **100 final usable
  SFT examples**, not merely 100 attempted prompts.
- If a candidate fails all retries, the pipeline creates a replacement
  candidate with the same scenario, difficulty, subject/topic, policy, split,
  coercion level, and guidance, but with a new id such as `dg_0101`.
- Replacement candidates still go through schema validation, local validation,
  and the LLM judge. They are not automatically accepted.
- If OpenAI rejects the LLM judge prompt, the candidate is marked with
  `llm_judge_prompt_rejected` and retried or replaced; the run does not crash.
- The run stops when 100 examples are accepted or when the 300-candidate cap is
  reached.

### Data Generation Setup

The data-generation scripts use the training/data stack, including the OpenAI
client and YAML parsing. Use the existing runtime venv if it already has those
dependencies, or install the training requirements in a separate environment:

```bash
python3 -m venv .venv-train
source .venv-train/bin/activate
pip install -U pip
pip install -r training/requirements-train.txt
```

Set the API key:

```bash
export OPENAI_API_KEY="sk-..."
```

### Generate 100 Examples

From the repo root:

```bash
python training/data_generation/run_pipeline.py \
  --config training/data_generation/configs/batch_001.yaml \
  --export-output data/generated_reviewed/batch_001 \
  --include-draft-valid \
  --convert-output training/data/sft.generated.batch_001.jsonl \
  --convert-stats training/data/sft.generated.batch_001.stats.json \
  --keep-going-on-validation-failure
```

During a run, accepted raw examples are written to:

```text
data/generated/batch_001/raw/
```

Monitor progress from another terminal:

```bash
find data/generated/batch_001/raw -maxdepth 1 -name '*.json' | wc -l
```

### Data Generation Outputs

| Output | Meaning |
| --- | --- |
| `data/generated/batch_001/generation_plan.json` | Initial and replacement candidate plan |
| `data/generated/batch_001/raw/` | Candidates accepted by schema, local validation, and LLM judge |
| `data/generated/batch_001/generation_rejected/` | Failed generation attempts with validation issues |
| `data/generated/batch_001/valid/` | Raw examples that pass final deterministic validation |
| `data/generated/batch_001/rejected/` | Raw examples rejected by final deterministic validation |
| `data/generated/batch_001/events.jsonl` | Structured event log |
| `data/generated/batch_001/run.log` | Human-readable log |
| `data/generated/batch_001/summary.json` | Latest stage summary |
| `data/generated_reviewed/batch_001/` | Exported examples for conversion |
| `training/data/sft.generated.batch_001.jsonl` | Final OpenAI-style SFT JSONL |
| `training/data/sft.generated.batch_001.stats.json` | SFT conversion stats |

For manual review workflows, omit `--include-draft-valid`; export will include
only examples marked `annotator.review_status: approved`.

## Training Smoke Pipeline

Use the training scripts when you want to prove the SFT loop is wired end to
end. Keep this in a separate venv because it installs heavier dependencies.

```bash
python3 -m venv .venv-train
source .venv-train/bin/activate
pip install -U pip
pip install -r training/requirements-train.txt
```

Convert hand-authored examples:

```bash
python training/convert.py
```

Train LoRA:

```bash
python training/train_sft.py --config training/configs/qwen2_5_0_5b.yaml
```

Merge the adapter for serving:

```bash
python training/merge_lora.py \
  --adapter out/qwen25-05b-tutor-lora-v0 \
  --out out/qwen25-05b-tutor-merged-v0
```

Run offline smoke checks:

```bash
python training/smoke_eval.py --skip-inference
```

See [`training/README.md`](training/README.md) for platform-specific commands
for macOS, Colab/Kaggle, and Linux GPU training.

## Tests

Install dev dependencies, then run:

```bash
pytest -q
```

or, using the project venv explicitly:

```bash
.venv/bin/python -m pytest
```

The focused data-generation tests are:

```bash
.venv/bin/python -m pytest tests/test_data_generation.py
```

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `OPENAI_API_KEY is not set` | Missing API key | Export `OPENAI_API_KEY` or add it to `.env` |
| LLM judge prompt rejected | Provider policy rejected the judge prompt | The pipeline now treats this as a rejected candidate and continues |
| Final JSONL has fewer than 100 rows | Generation hit the safety cap or export excluded drafts | Check `summary.json`, use `--include-draft-valid` for smoke runs, or increase `max_candidate_examples` |
| vLLM connection refused | Server not running or wrong base URL | Confirm `http://host:port/v1` and `COGNISHIELD_OPENAI_API_BASE` |
| Structured output parse errors | Backend/model does not reliably support JSON/tool-style output | Try a stronger instruct model, lower temperature, or OpenAI backend |

## More Documentation

- [`docs/architecture.md`](docs/architecture.md) - runtime pipeline diagrams and prompt details
- [`docs/modules.md`](docs/modules.md) - module reference
- [`docs/vllm-inference.md`](docs/vllm-inference.md) - vLLM server and inference setup
- [`docs/annotation_guidelines.md`](docs/annotation_guidelines.md) - data annotation and SFT format guidance
- [`training/README.md`](training/README.md) - SFT training, merge, and eval workflow
- [`pipeline.txt`](pipeline.txt) - original phase-1 design sketch

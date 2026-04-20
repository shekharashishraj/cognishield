# CogniBench data generation script

[`cognibench_pipeline.py`](../cognibench_pipeline.py) is a **standalone** script. It does **not** import the `cognishield` package.

## Purpose

Generates synthetic **6-turn** student–tutor conversations as **JSONL** (one JSON object per line). Each row includes:

- `split` / `split_label` (exemplary legitimate, adequate ambiguous, failing disallowed)
- STEM `subject`, `problem`, `expected_behavior`, `coercion_level`
- `turns`: list of `{role, content, turn_number}`

## Dependencies

```bash
pip install anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

The script hard-codes `MODEL = "claude-haiku-4-5-20251001"` (see top of file).

## Run

From repo root:

```bash
python3 cognibench_pipeline.py --n 2 --output cognibench.jsonl
```

Useful flags:

| Flag | Meaning |
|------|---------|
| `--n` | Conversations per split **per topic** (default 5) |
| `--output` | JSONL path (default `cognibench.jsonl`) |
| `--topics` | Space-separated subject names to filter (must match `TOPICS` entries) |
| `--splits` | Subset of `exemplary_legitimate`, `adequate_ambiguous`, `failing_disallowed` |
| `--preview-only` | Print samples from existing JSONL without calling the API |
| `--preview-n` | How many conversations to preview |

After a normal run, the script previews `--preview-n` conversations from the output file.

## Relation to CogniShield

Use CogniBench output as **evaluation or training data** for the shielding pipeline; run CogniShield separately per user message (see [vllm-inference.md](vllm-inference.md) or root [README.md](../README.md)).

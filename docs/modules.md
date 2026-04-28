# Module reference

Paths are relative to the repository root.

## Package `cognishield`

### `cognishield/app/schemas.py`

Pydantic models shared across chains and the verifier:

- `PlannerOutput`, `GeneratorOutput`, `ValidatorOutput`, `VerifierDecision`
- Meta pipeline: `MetaAgentOutput`, `RevisionOutput`, plus concern/answer level types — see `schemas.py`
- `TurnContext`, `CogniShieldState` (meta pipeline fields: `primary_draft`, `meta_output`, `meta_verifier_decision`, `final_response_text`)

Structured outputs from the LLM must validate against these models.

### `cognishield/app/settings.py`

`Settings(BaseSettings)`: single configuration object.

- **Env prefix:** `COGNISHIELD_` (plus optional `.env` in cwd)
- **LLM:** `model`, `openai_api_base`, `openai_api_key` (falls back to `OPENAI_API_KEY` when key unset in settings)
- **Pipeline:** `pipeline` — `legacy` (default) or `meta` (primary → meta-agent → revision). See [architecture.md](architecture.md).
- **Temps:** `temperature_planner`, `temperature_generator`, `temperature_validators`; meta path also uses `temperature_meta`, `temperature_revision`
- **Loop:** `max_revisions` (legacy path only; meta path does not loop)
- **Validators (legacy):** `enable_bloom`, `enable_cognitive`, `enable_safety`, `enable_accuracy` (disabled validators inject passing stubs)
- **Verifier floors (legacy):** `verifier_accuracy_min_score`, `verifier_cognitive_min_score`, `verifier_bloom_min_score`
- **Meta verifier:** `meta_verifier_max_cognitive_concern`, `meta_verifier_max_safety_concern`, `meta_verifier_min_answer_quality`
- **Logging:** `log_level`, `log_file`
- **Trace:** `trace_path`, `trace_stdout`, `trace_redact`
- **CLI:** `query` (also `COGNISHIELD_QUERY`)
- **Debug:** `dry_run` (no LLM calls; stub outputs)

### `cognishield/app/cli.py`

`main()` → `tyro.cli(Settings)` → `configure_logging` → `init_state` → `run_turn` → print final answer to **stdout**, banner to **stderr**.

### `cognishield/app/orchestrator.py`

`run_turn(state, settings, tracer=None) -> str`

- Builds chains from `settings` each run (no global cached chains).
- Serializes dict fields to JSON strings for prompt templates.
- Emits trace events:
  - **Legacy:** `plan`, `generate`, `validate:<name>`, `verify`, `accept` / `revise`, `max_revisions`.
  - **Meta:** `primary`, `meta`, `verify`, `revision`.

### `cognishield/app/verifier.py`

- **Legacy:** `verify_with_rules(plan, candidate, reports, settings)` — safety booleans (`answer_leakage`, `direct_solution`, `prompt_injection_detected`); validator scores vs `Settings` thresholds. Single backprompt text on revise.
- **Meta pipeline:** `verify_meta_classifiers(meta, settings)` — ordinal thresholds on structured meta classifiers; no LLM.

### `cognishield/app/state.py`

`init_state(user_query, ...)` constructs `CogniShieldState` with a `TurnContext`.

### `cognishield/app/policies.py`

Static taxonomy and policy snippets (`INTERVENTIONS`, `DISALLOWED_PATTERNS`, `DEFAULT_POLICY`). Used for documentation / future hooks; planner prompt encodes behavior today.

### `cognishield/app/config.py`

Thin `get_settings()` re-export for compatibility.

### `cognishield/app/logging_setup.py`

`configure_logging(settings)` — root logger, optional file handler, quiets noisy HTTP libraries below DEBUG.

### `cognishield/app/trace.py`

`JsonlTracer`: writes JSON lines when `trace_path` and/or `trace_stdout` (stderr stream) is enabled.

### `cognishield/app/chains/llm_factory.py`

`planner_llm`, `generator_llm`, `validator_llm`; **meta pipeline:** `meta_llm`, `revision_llm` — each returns `ChatOpenAI(settings, role_temperature)`.

### `cognishield/app/chains/planner_chain.py` (and `generator_chain.py`, `validator_chains.py`)

`build_*_chain(settings)` → `Runnable` (`ChatPromptTemplate | llm.with_structured_output(...)`).

### `cognishield/app/chains/primary_chain.py`, `meta_chain.py`, `revision_chain.py`

Used when `pipeline=meta`. Same Runnable pattern; prompts in `prompts/primary.txt`, `meta_agent.txt`, `revision.txt`.

Prompt files for all chains live in `cognishield/app/prompts/`.

### `cognishield/app/main.py` / `cognishield/app/__main__.py`

Delegate to CLI `main()`.

---

## Tests (`tests/`)

| File | Purpose |
|------|---------|
| `test_verifier.py` | Rule logic with synthetic `ValidatorOutput` |
| `test_cli_help.py` | Subprocess `python -m cognishield.app.cli --help` |
| `test_orchestrator_dry_run.py` | `run_turn` with `dry_run=True` (no network) |

---

## Root scripts

### `cognibench_pipeline.py`

Anthropic-based JSONL generator for CogniBench-style conversations. Constants: `MODEL`, `TOPICS`, `SPLITS`, `OUTPUT_FILE`. CLI: `--n`, `--output`, `--preview-only`, `--topics`, `--splits`.

Requires `ANTHROPIC_API_KEY` and `anthropic` package (not in core `cognishield` dependencies).

### `pipeline.txt`

Human-readable design sketch for Phase 1; not executed by the runtime.

---

## Build metadata

- [`pyproject.toml`](../pyproject.toml): package name, dependencies, `package-data` for `prompts/*.txt`, pytest config.
- [`requirements.txt`](../requirements.txt): pinned-style list for non-editable installs.

# Module reference

Paths are relative to the repository root.

## Package `cognishield`

### `cognishield/app/schemas.py`

Pydantic models shared across chains and the verifier:

- `PlannerOutput`, `GeneratorOutput`, `ValidatorOutput`, `VerifierDecision`
- `TurnContext`, `CogniShieldState`

Structured outputs from the LLM must validate against these models.

### `cognishield/app/settings.py`

`Settings(BaseSettings)`: single configuration object.

- **Env prefix:** `COGNISHIELD_` (plus optional `.env` in cwd)
- **LLM:** `model`, `openai_api_base`, `openai_api_key` (falls back to `OPENAI_API_KEY` when key unset in settings)
- **Temps:** `temperature_planner`, `temperature_generator`, `temperature_validators`
- **Loop:** `max_revisions`
- **Validators:** `enable_bloom`, `enable_cognitive`, `enable_safety`, `enable_accuracy` (disabled validators inject passing stubs)
- **Verifier floors:** `verifier_accuracy_min_score`, `verifier_cognitive_min_score`, `verifier_bloom_min_score`
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
- Emits trace events: `plan`, `generate`, `validate:<name>`, `verify`, `accept` / `revise`, `max_revisions`.

### `cognishield/app/verifier.py`

`verify_with_rules(plan, candidate, reports, settings)`

Rules: safety booleans (`answer_leakage`, `direct_solution`, `prompt_injection_detected`); validator scores vs `Settings` thresholds. Single backprompt text on revise.

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

`planner_llm`, `generator_llm`, `validator_llm` — each returns `ChatOpenAI(settings, role_temperature)`.

### `cognishield/app/chains/planner_chain.py` (and `generator_chain.py`, `validator_chains.py`)

`build_*_chain(settings)` → `Runnable` (`ChatPromptTemplate | llm.with_structured_output(...)`).

Prompt files live in `cognishield/app/prompts/`.

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

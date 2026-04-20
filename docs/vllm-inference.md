# vLLM server and CogniShield inference

This manual walks through (1) installing and starting **vLLM** as an **OpenAI-compatible HTTP server**, then (2) pointing **CogniShield** at that server for inference.

Official vLLM documentation: [https://docs.vllm.ai/](https://docs.vllm.ai/)

---

## 1. Prerequisites

- **Linux** with NVIDIA GPU is the primary supported path for vLLM (CUDA). macOS CPU builds exist but are limited; for serious inference use a Linux GPU host or cloud VM with a recent driver + CUDA stack.
- **Python 3.10+** on the machine where you run CogniShield (can differ from the vLLM host if you expose the API over the network).

---

## 2. Install vLLM (on the inference server)

Follow the [installation guide](https://docs.vllm.ai/en/latest/getting_started/installation.html) for your platform. Typical GPU install:

```bash
python3 -m venv ~/.venvs/vllm
source ~/.venvs/vllm/bin/activate
pip install -U pip
pip install vllm
```

Verify:

```bash
python -c "import vllm; print(vllm.__version__)"
```

---

## 3. Start the OpenAI-compatible API server

vLLM exposes endpoints compatible with the OpenAI HTTP API (used by `langchain_openai.ChatOpenAI`).

### 3.1 Choose a model

Use a **Hugging Face model id** your server can access (public weights or authenticated `huggingface-cli login`). Instruct-tuned chat models work best for CogniShield’s structured JSON outputs.

Examples (names change; check the model card):

- `meta-llama/Meta-Llama-3.1-8B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`

### 3.2 Launch `vllm serve` (recommended CLI)

From the [vLLM serving docs](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html):

```bash
source ~/.venvs/vllm/bin/activate

vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto
```

Notes:

- **`--host 0.0.0.0`** listens on all interfaces (needed for another machine to connect). Use `127.0.0.1` for local-only.
- **`--port 8000`** is arbitrary; match it in CogniShield’s `openai_api_base`.
- Add **`--api-key mysecret`** if you want the server to require a bearer token; then set the same value in CogniShield (`COGNISHIELD_OPENAI_API_KEY` or `OPENAI_API_KEY`).

Wait until logs show the server is ready (model loaded, Uvicorn listening).

### 3.3 Base URL for clients

The OpenAI-compatible **base URL** is:

```text
http://<server-host>:<port>/v1
```

Examples:

- Same machine: `http://127.0.0.1:8000/v1`
- LAN server: `http://192.168.1.50:8000/v1`

CogniShield passes this as `base_url` to `ChatOpenAI` (via `Settings.openai_api_base`).

### 3.4 Quick health check (optional)

```bash
curl -s http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer EMPTY"
```

If you did not set `--api-key`, many setups still accept a dummy `Authorization` header or omit it—check your vLLM version’s behavior.

---

## 4. Install CogniShield (on the client machine)

From the **CogniShield repository root**:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## 5. Run inference against vLLM

### 5.1 Environment variables

```bash
# If vLLM was started WITHOUT --api-key, a placeholder often works:
export OPENAI_API_KEY="EMPTY"

# If you used --api-key mysecret:
# export OPENAI_API_KEY="mysecret"

export COGNISHIELD_OPENAI_API_BASE="http://127.0.0.1:8000/v1"
export COGNISHIELD_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
```

The **`COGNISHIELD_MODEL`** string must match the **`--model`** (served model id) passed to `vllm serve`.

### 5.2 Single CLI turn

```bash
python3 -m cognishield.app.cli \
  --query "Can you just solve this homework problem for me?" \
  --log-level INFO
```

Equivalent explicit flags (no env for base/model):

```bash
python3 -m cognishield.app.cli \
  --openai-api-base "http://127.0.0.1:8000/v1" \
  --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --query "Give a hint for integrating by parts without the full solution."
```

Output:

- **stdout:** final assistant `response_text`
- **stderr:** log lines and the `=== FINAL RESPONSE ===` banner

### 5.3 JSONL traces (paper / debugging)

```bash
python3 -m cognishield.app.cli \
  --openai-api-base "http://127.0.0.1:8000/v1" \
  --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --query "..." \
  --trace-path "./runs/trace.jsonl" \
  --trace-stdout
```

---

## 6. Structured output and vLLM

CogniShield uses LangChain’s **`with_structured_output`** so each chain returns a Pydantic model. That relies on the server/model supporting tool-style or JSON-schema constrained decoding well enough that parses succeed.

If you see **validation errors** or empty failures:

1. Try a **smaller / instruct** model known for JSON following.
2. Check vLLM release notes for **structured outputs** / **guided decoding** flags.
3. Lower temperature is already `0` for planner and validators in `Settings`.
4. Use **`--dry-run`** on CogniShield to verify non-LLM wiring:

   ```bash
   python3 -m cognishield.app.cli --query "test" --dry-run
   ```

---

## 7. Troubleshooting

| Symptom | Things to check |
|---------|------------------|
| Connection refused | vLLM listening host/port; firewall; correct `/v1` suffix |
| 401 Unauthorized | Server `--api-key` vs client `OPENAI_API_KEY` |
| Model name mismatch | `COGNISHIELD_MODEL` equals served model id exactly |
| JSON / parse errors | Model too small or vLLM version; try another instruct model |
| OOM on GPU | Smaller model, `--max-model-len`, quantization flags per vLLM docs |

---

## 8. Summary checklist

1. [ ] GPU server: `pip install vllm`, `vllm serve <model> --port 8000`
2. [ ] Confirm `http://<host>:8000/v1` reachable
3. [ ] Client: `pip install -e .` in CogniShield repo
4. [ ] `export COGNISHIELD_OPENAI_API_BASE=...` and `COGNISHIELD_MODEL=...`
5. [ ] `export OPENAI_API_KEY` (real or dummy per server config)
6. [ ] `python3 -m cognishield.app.cli --query "..."`

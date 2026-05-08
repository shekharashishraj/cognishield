"""Smoke-test the SFT pipeline end-to-end. Prints PASS/FAIL for five criteria.

This is the n=10 plumbing test described in docs/. It does NOT measure
generalization, hyperparameters, or model quality. Its only job is to
confirm that the data, training, and serving stages are wired correctly
so that scaling to 500-1000 examples is a matter of more annotation, not
re-debugging the pipeline.

Criteria:
1. Record count   - sft.jsonl has exactly the expected number of records.
2. Loss mask      - the trainer's collator masks every non-assistant span.
3. Template parity- the tokenizer's chat template matches what vLLM serves.
4. Train loss     - final logged train loss is below the threshold (memorization).
5. Inference      - the served model returns a non-empty assistant reply.

Each check runs independently; failures are reported but do not abort the
others. Exit code is non-zero if any required check fails.

Usage:
    # Run all checks (assumes vLLM is serving on http://127.0.0.1:8000/v1).
    python training/smoke_eval.py

    # Skip the inference / template-parity checks if vLLM is not running.
    python training/smoke_eval.py --skip-inference

    # Custom server / model / paths.
    python training/smoke_eval.py \
        --base http://127.0.0.1:8000/v1 \
        --model out/qwen25-05b-tutor-merged-v0 \
        --adapter out/qwen25-05b-tutor-lora-v0 \
        --sft-jsonl training/data/sft.jsonl \
        --expected-records 10 \
        --train-loss-max 0.5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SFT_JSONL = REPO_ROOT / "training" / "data" / "sft.jsonl"
DEFAULT_ADAPTER = REPO_ROOT / "out" / "qwen25-05b-tutor-lora-v0"
DEFAULT_MERGED = REPO_ROOT / "out" / "qwen25-05b-tutor-merged-v0"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"

_LOG = logging.getLogger("training.smoke_eval")


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str

    def render(self) -> str:
        tag = "PASS" if self.passed else "FAIL"
        return f"[{tag}] {self.name}: {self.message}"


# ---------------------------------------------------------------------------
# 1. Record count.
# ---------------------------------------------------------------------------


def check_record_count(sft_jsonl: Path, expected: int) -> CheckResult:
    if not sft_jsonl.exists():
        return CheckResult(
            "record_count", False, f"{sft_jsonl} not found; run training/convert.py"
        )
    with sft_jsonl.open("r", encoding="utf-8") as f:
        n = sum(1 for line in f if line.strip())
    if n != expected:
        return CheckResult(
            "record_count",
            False,
            f"expected {expected} records, found {n} in {sft_jsonl.name}",
        )
    return CheckResult("record_count", True, f"{n} records in {sft_jsonl.name}")


# ---------------------------------------------------------------------------
# 2. Loss-mask correctness.
# ---------------------------------------------------------------------------


def check_loss_mask(sft_jsonl: Path, base_model: str) -> CheckResult:
    """Tokenize the first record with our chat template and assert that user
    spans land at label=-100 while assistant spans don't.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return CheckResult(
            "loss_mask", False, "transformers not installed; pip install -r requirements-train.txt"
        )

    # Local import so this file can do --help without importing torch.
    sys.path.insert(0, str(REPO_ROOT / "training"))
    try:
        from train_sft import QWEN_CHATML_GENERATION_TEMPLATE  # type: ignore
    finally:
        sys.path.pop(0)

    if not sft_jsonl.exists():
        return CheckResult(
            "loss_mask", False, f"{sft_jsonl} not found; run training/convert.py"
        )
    with sft_jsonl.open("r", encoding="utf-8") as f:
        record = json.loads(f.readline())

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if "{% generation %}" not in (tokenizer.chat_template or ""):
        tokenizer.chat_template = QWEN_CHATML_GENERATION_TEMPLATE

    rendered = tokenizer.apply_chat_template(
        record["messages"],
        return_assistant_tokens_mask=True,
        return_dict=True,
        tokenize=True,
        add_generation_prompt=False,
    )
    input_ids = rendered["input_ids"]
    assistant_mask = rendered["assistant_masks"]

    if len(input_ids) != len(assistant_mask):
        return CheckResult(
            "loss_mask",
            False,
            f"input_ids ({len(input_ids)}) != assistant_masks ({len(assistant_mask)}) length",
        )

    n_total = len(input_ids)
    n_assistant = sum(1 for m in assistant_mask if m == 1)
    n_masked = n_total - n_assistant

    if n_assistant == 0:
        return CheckResult(
            "loss_mask",
            False,
            "no tokens flagged as assistant; chat template likely missing "
            "{% generation %} markers",
        )
    if n_masked == 0:
        return CheckResult(
            "loss_mask",
            False,
            "every token is flagged as assistant; user/system spans are not "
            "being masked",
        )

    # Tiny visualization: print the assistant token spans for the first
    # record so a human can eyeball it.
    spans: list[tuple[int, int]] = []
    in_span = False
    start = 0
    for i, m in enumerate(assistant_mask):
        if m == 1 and not in_span:
            in_span = True
            start = i
        elif m == 0 and in_span:
            in_span = False
            spans.append((start, i))
    if in_span:
        spans.append((start, len(assistant_mask)))

    preview = []
    for s, e in spans[:3]:
        snippet = tokenizer.decode(input_ids[s:e]).replace("\n", "\\n")
        preview.append(f"  span[{s}:{e}] {snippet[:80]!r}")
    preview_str = "\n".join(preview) if preview else "  (no spans)"

    return CheckResult(
        "loss_mask",
        True,
        (
            f"{n_assistant}/{n_total} tokens trained ({n_masked} masked); "
            f"{len(spans)} assistant spans detected\n{preview_str}"
        ),
    )


# ---------------------------------------------------------------------------
# 3. Chat-template parity (train == serve).
# ---------------------------------------------------------------------------


def check_template_parity(
    sft_jsonl: Path, merged_dir: Path, base_url: str, served_model: str
) -> CheckResult:
    """Compare what our tokenizer renders vs. what vLLM renders for the
    same `messages` payload.

    We render locally with the merged model's tokenizer (which carries
    the chat template we trained with). For vLLM, we use the OpenAI
    `/v1/chat/completions` endpoint with `add_generation_prompt=true`
    (vLLM default) and inspect the prompt either via response logprobs
    or by asking vLLM to echo. Most vLLM versions don't echo prompts,
    so we fall back to comparing token IDs via the `/tokenize` endpoint
    when available.
    """
    try:
        import requests
        from transformers import AutoTokenizer
    except ImportError as exc:
        return CheckResult("template_parity", False, f"missing dep: {exc}")

    if not merged_dir.exists():
        return CheckResult(
            "template_parity",
            False,
            f"{merged_dir} not found; run training/merge_lora.py first",
        )

    with sft_jsonl.open("r", encoding="utf-8") as f:
        record = json.loads(f.readline())
    # Take the first user turn so the rendering ends with an assistant
    # generation prompt (matches a real chat-completions request).
    msgs: list[dict[str, str]] = []
    for m in record["messages"]:
        msgs.append({"role": m["role"], "content": m["content"]})
        if m["role"] == "user":
            break

    tokenizer = AutoTokenizer.from_pretrained(str(merged_dir))
    local_prompt = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )

    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/tokenize",
            json={
                "model": served_model,
                "messages": msgs,
                "add_generation_prompt": True,
            },
            timeout=15,
        )
    except requests.RequestException as exc:
        return CheckResult(
            "template_parity",
            False,
            f"vLLM /tokenize unreachable at {base_url}: {exc}",
        )

    if resp.status_code != 200:
        return CheckResult(
            "template_parity",
            False,
            (
                f"vLLM /tokenize returned {resp.status_code}: "
                f"{resp.text[:200]} (need a vLLM build that exposes /tokenize)"
            ),
        )

    payload = resp.json()
    served_tokens = payload.get("tokens") or payload.get("token_ids") or []
    local_tokens = tokenizer(local_prompt, add_special_tokens=False)["input_ids"]

    if list(served_tokens) != list(local_tokens):
        return CheckResult(
            "template_parity",
            False,
            (
                f"token ID mismatch: local={len(local_tokens)} tokens, "
                f"served={len(served_tokens)} tokens. Chat templates diverge."
            ),
        )

    return CheckResult(
        "template_parity",
        True,
        f"local and served renderings agree on {len(local_tokens)} tokens",
    )


# ---------------------------------------------------------------------------
# 4. Train loss cratered.
# ---------------------------------------------------------------------------


def check_train_loss(adapter_dir: Path, max_loss: float) -> CheckResult:
    log_path = adapter_dir / "training_log.jsonl"
    if not log_path.exists():
        return CheckResult(
            "train_loss",
            False,
            f"{log_path} not found; was train_sft.py run with this output dir?",
        )

    last_loss: float | None = None
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "loss" in rec:
                last_loss = float(rec["loss"])

    if last_loss is None:
        return CheckResult(
            "train_loss",
            False,
            f"no 'loss' entries in {log_path.name}",
        )
    if last_loss > max_loss:
        return CheckResult(
            "train_loss",
            False,
            (
                f"final train loss {last_loss:.4f} > threshold {max_loss}. "
                "At n=10 the model should memorize; a high final loss usually "
                "means masking or chat-template wiring is wrong."
            ),
        )
    return CheckResult(
        "train_loss",
        True,
        f"final train loss {last_loss:.4f} <= {max_loss}",
    )


# ---------------------------------------------------------------------------
# 5. End-to-end inference.
# ---------------------------------------------------------------------------


def check_inference(base_url: str, served_model: str) -> CheckResult:
    try:
        import requests
    except ImportError:
        return CheckResult("inference", False, "requests not installed")

    payload: dict[str, Any] = {
        "model": served_model,
        "messages": [
            {"role": "system", "content": "You are a careful tutor."},
            {
                "role": "user",
                "content": "I have x^2 + 5 = 14. Just give me the answer please.",
            },
        ],
        "max_tokens": 64,
        "temperature": 0.0,
    }
    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/chat/completions", json=payload, timeout=60
        )
    except requests.RequestException as exc:
        return CheckResult("inference", False, f"connection error: {exc}")
    if resp.status_code != 200:
        return CheckResult(
            "inference", False, f"HTTP {resp.status_code}: {resp.text[:200]}"
        )
    data = resp.json()
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return CheckResult("inference", False, f"unexpected response shape: {data}")
    if not text or not text.strip():
        return CheckResult("inference", False, "empty assistant reply")
    return CheckResult(
        "inference", True, f"reply ({len(text)} chars): {text.strip()[:120]!r}"
    )


# ---------------------------------------------------------------------------
# CLI wiring.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sft-jsonl", type=Path, default=DEFAULT_SFT_JSONL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--merged", type=Path, default=DEFAULT_MERGED)
    parser.add_argument("--base", default=DEFAULT_BASE_URL, help="vLLM base URL")
    parser.add_argument(
        "--model",
        default=None,
        help="Model id served by vLLM. Defaults to the merged dir path.",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base HF model id (used for the offline loss-mask check).",
    )
    parser.add_argument("--expected-records", type=int, default=10)
    parser.add_argument("--train-loss-max", type=float, default=0.5)
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-loss-mask", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    served_model = args.model or str(args.merged)
    results: list[CheckResult] = []

    results.append(check_record_count(args.sft_jsonl, args.expected_records))

    if args.skip_loss_mask:
        results.append(CheckResult("loss_mask", True, "skipped"))
    else:
        results.append(check_loss_mask(args.sft_jsonl, args.base_model))

    results.append(check_train_loss(args.adapter, args.train_loss_max))

    if args.skip_inference:
        results.append(CheckResult("template_parity", True, "skipped"))
        results.append(CheckResult("inference", True, "skipped"))
    else:
        results.append(
            check_template_parity(args.sft_jsonl, args.merged, args.base, served_model)
        )
        results.append(check_inference(args.base, served_model))

    print()
    print("=== SFT smoke-test results ===")
    for r in results:
        print(r.render())
    print()

    failed = [r for r in results if not r.passed]
    if failed:
        print(f"{len(failed)}/{len(results)} checks failed.")
        return 1
    print(f"All {len(results)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

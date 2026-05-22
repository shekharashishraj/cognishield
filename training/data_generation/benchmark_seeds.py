"""Sample benchmark problems + reference solutions for data generation.

HF dataset IDs used (see loaders below). **We do not use** ``hendrycks/competition_math``
(Hub instability / access issues).

Current sources:
- GSM8K: ``openai/gsm8k`` (config ``main``, split ``train``)
- SVAMP: ``ChilleD/SVAMP`` (split ``train``)
- MATH (high school + Level 4--5 filter): ``EleutherAI/hendrycks_math`` — all topic configs’
  ``train`` splits concatenated (replaces inaccessible ``EleutherAI/MATH`` on the Hub).
- MATH-500 (undergrad alternate): ``HuggingFaceH4/MATH-500`` (split ``test``)
- Hendrycks MATH single-topic (undergrad alternate): random ``EleutherAI/hendrycks_math/{subset}/train``
- MBPP: ``google-research-datasets/mbpp`` (config ``sanitized``, split ``train``)
- HumanEval: ``openai/openai_humaneval`` (split ``test``)
- Undergraduate coding: **no APPS** — ``codeparrot/apps`` fails on modern ``datasets`` (dataset scripts);
  we alternate **MBPP** and **HumanEval** (same loaders as lower tiers).
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from typing import Any

from datasets import Dataset, concatenate_datasets, load_dataset

# Optional caps (characters) to avoid blowing LLM context on huge items.
MAX_PROBLEM_CHARS = 16000
MAX_REFERENCE_CHARS = 24000

HENDRYCKS_MATH_SUBSETS: tuple[str, ...] = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


@dataclass(frozen=True)
class BenchmarkSeed:
    problem_statement: str
    reference_solution: str
    seed_dataset: str
    seed_example_id: str
    subject: str
    topic: str


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 24].rstrip() + "\n...[truncated]"


_cache: dict[str, Dataset] = {}


def _get_cached(name: str, loader) -> Dataset:
    if name not in _cache:
        _cache[name] = loader()
    return _cache[name]


def _load_gsm8k_train() -> Dataset:
    return load_dataset("openai/gsm8k", "main", split="train")


def _load_svamp_train() -> Dataset:
    return load_dataset("ChilleD/SVAMP", split="train")


MATH_TRAIN_CACHE_KEY = "EleutherAI_hendrycks_math_concat_train"


def _load_math_train() -> Dataset:
    """Full MATH-style train pool: concat all ``EleutherAI/hendrycks_math`` topic trains.

    ``EleutherAI/MATH`` is often ``DatasetNotFoundError`` on the Hub; this mirror matches
    the same ``problem`` / ``solution`` / ``level`` / ``type`` schema.
    """
    parts: list[Dataset] = []
    for cfg in HENDRYCKS_MATH_SUBSETS:
        parts.append(load_dataset("EleutherAI/hendrycks_math", cfg, split="train"))
    return concatenate_datasets(parts)


def _load_math500_test() -> Dataset:
    return load_dataset("HuggingFaceH4/MATH-500", split="test")


def _load_hendrycks_math_train(config: str) -> Dataset:
    return load_dataset("EleutherAI/hendrycks_math", config, split="train")


def _load_math_hard_subset() -> Dataset:
    """Level 4--5 only over the concatenated hendrycks_math train pool."""
    key = "EleutherAI_hendrycks_math_concat_train_level_4_5"
    if key not in _cache:
        full = _get_cached(MATH_TRAIN_CACHE_KEY, _load_math_train)

        def keep(row: dict[str, Any]) -> bool:
            lev = str(row.get("level", ""))
            return lev in {"Level 4", "Level 5"}

        _cache[key] = full.filter(keep)
    return _cache[key]


def _load_mbpp_train() -> Dataset:
    return load_dataset("google-research-datasets/mbpp", "sanitized", split="train")


def _load_humaneval_test() -> Dataset:
    return load_dataset("openai/openai_humaneval", split="test")


def _pick_row(ds: Dataset, rng: random.Random) -> tuple[int, dict[str, Any]]:
    idx = rng.randrange(len(ds))
    return idx, ds[idx]


def _math_subject_topic(row: dict[str, Any], difficulty: str, rng: random.Random) -> tuple[str, str]:
    from training.data_generation.taxonomy import MATH_TOPICS

    typ = row.get("type")
    if not (isinstance(typ, str) and typ.strip()):
        subj_field = row.get("subject")
        if isinstance(subj_field, str) and subj_field.strip():
            typ = subj_field.strip()
    if isinstance(typ, str) and typ.strip():
        return "Mathematics", typ.strip()
    return rng.choice(MATH_TOPICS[difficulty])


def _coding_subject_topic(difficulty: str, rng: random.Random) -> tuple[str, str]:
    from training.data_generation.taxonomy import CODING_TOPICS

    return rng.choice(CODING_TOPICS[difficulty])


def _benchmark_seed_from_math_row(
    row: dict[str, Any],
    rng: random.Random,
    difficulty: str,
    *,
    seed_dataset: str,
    seed_example_id: str,
) -> BenchmarkSeed:
    prob = _truncate(str(row["problem"]), MAX_PROBLEM_CHARS)
    ref = _truncate(str(row["solution"]), MAX_REFERENCE_CHARS)
    subj, topic = _math_subject_topic(row, difficulty, rng)
    return BenchmarkSeed(
        problem_statement=prob,
        reference_solution=ref,
        seed_dataset=seed_dataset,
        seed_example_id=seed_example_id,
        subject=subj,
        topic=topic,
    )


def sample_gsm8k(rng: random.Random, difficulty: str) -> BenchmarkSeed:
    ds = _get_cached("openai_gsm8k_main_train", _load_gsm8k_train)
    idx, row = _pick_row(ds, rng)
    prob = _truncate(str(row["question"]), MAX_PROBLEM_CHARS)
    ref = _truncate(str(row["answer"]), MAX_REFERENCE_CHARS)
    subj, topic = _math_subject_topic(row, difficulty, rng)
    return BenchmarkSeed(
        problem_statement=prob,
        reference_solution=ref,
        seed_dataset="openai/gsm8k/main/train",
        seed_example_id=str(idx),
        subject=subj,
        topic=topic,
    )


def sample_svamp(rng: random.Random, difficulty: str) -> BenchmarkSeed:
    ds = _get_cached("ChilleD/SVAMP_train", _load_svamp_train)
    idx, row = _pick_row(ds, rng)
    body = str(row.get("Body", "")).strip()
    q = str(row.get("Question", "")).strip()
    eq = str(row.get("Equation", "")).strip()
    parts = [p for p in (body, q, eq) if p]
    prob = _truncate("\n".join(parts), MAX_PROBLEM_CHARS)
    ref = _truncate(str(row.get("Answer", "")), MAX_REFERENCE_CHARS)
    subj, topic = _math_subject_topic({}, difficulty, rng)
    return BenchmarkSeed(
        problem_statement=prob,
        reference_solution=ref,
        seed_dataset="ChilleD/SVAMP/train",
        seed_example_id=str(idx),
        subject=subj,
        topic=topic,
    )


def sample_math_train_row(rng: random.Random, difficulty: str) -> BenchmarkSeed:
    ds = _get_cached(MATH_TRAIN_CACHE_KEY, _load_math_train)
    idx, row = _pick_row(ds, rng)
    return _benchmark_seed_from_math_row(
        row,
        rng,
        difficulty,
        seed_dataset="EleutherAI/hendrycks_math/concat/train",
        seed_example_id=str(idx),
    )


def _sample_undergrad_alternative(rng: random.Random, difficulty: str) -> BenchmarkSeed:
    """Try MATH-500, then EleutherAI/hendrycks_math, then Level 4--5 MATH only."""
    try:
        ds = _get_cached("HuggingFaceH4/MATH-500_test", _load_math500_test)
        if len(ds) > 0:
            idx, row = _pick_row(ds, rng)
            uid = str(row.get("unique_id", idx))
            return _benchmark_seed_from_math_row(
                row,
                rng,
                difficulty,
                seed_dataset="HuggingFaceH4/MATH-500/test",
                seed_example_id=uid,
            )
    except Exception as exc:
        warnings.warn(
            f"HuggingFaceH4/MATH-500 unavailable ({exc}); trying EleutherAI/hendrycks_math.",
            UserWarning,
            stacklevel=2,
        )

    config = rng.choice(HENDRYCKS_MATH_SUBSETS)
    try:
        ds = _get_cached(
            f"EleutherAI/hendrycks_math_{config}_train",
            lambda c=config: _load_hendrycks_math_train(c),
        )
        if len(ds) > 0:
            idx, row = _pick_row(ds, rng)
            return _benchmark_seed_from_math_row(
                row,
                rng,
                difficulty,
                seed_dataset=f"EleutherAI/hendrycks_math/{config}/train",
                seed_example_id=str(idx),
            )
    except Exception as exc:
        warnings.warn(
            f"EleutherAI/hendrycks_math/{config} unavailable ({exc}); "
            "using Level 4--5 pool from concatenated hendrycks_math train only.",
            UserWarning,
            stacklevel=2,
        )

    ds = _load_math_hard_subset()
    if len(ds) == 0:
        raise RuntimeError(
            "No undergraduate math seeds: Level 4--5 subset empty and alternatives failed."
        )
    warnings.warn(
        "Using Level 4--5 hendrycks_math concat train only as undergraduate math fallback.",
        UserWarning,
        stacklevel=2,
    )
    idx, row = _pick_row(ds, rng)
    return _benchmark_seed_from_math_row(
        row,
        rng,
        difficulty,
        seed_dataset="EleutherAI/hendrycks_math/concat/train/Level_4_5",
        seed_example_id=str(idx),
    )


def sample_math_undergrad(rng: random.Random, difficulty: str) -> BenchmarkSeed:
    if rng.random() < 0.5:
        ds = _load_math_hard_subset()
        if len(ds) == 0:
            warnings.warn(
                "hendrycks_math Level 4--5 subset empty; using alternate undergrad pools.",
                UserWarning,
                stacklevel=2,
            )
            return _sample_undergrad_alternative(rng, difficulty)
        idx, row = _pick_row(ds, rng)
        return _benchmark_seed_from_math_row(
            row,
            rng,
            difficulty,
            seed_dataset="EleutherAI/hendrycks_math/concat/train/Level_4_5",
            seed_example_id=str(idx),
        )
    return _sample_undergrad_alternative(rng, difficulty)


def sample_mbpp(rng: random.Random, difficulty: str) -> BenchmarkSeed:
    ds = _get_cached("google_research_mbpp_sanitized_train", _load_mbpp_train)
    idx, row = _pick_row(ds, rng)
    text = str(row.get("text") or row.get("prompt") or "").strip()
    code = str(row.get("code", ""))
    prob = _truncate(text, MAX_PROBLEM_CHARS)
    ref = _truncate(code, MAX_REFERENCE_CHARS)
    subj, topic = _coding_subject_topic(difficulty, rng)
    tid = row.get("task_id", idx)
    return BenchmarkSeed(
        problem_statement=prob,
        reference_solution=ref,
        seed_dataset="google-research-datasets/mbpp/sanitized/train",
        seed_example_id=str(tid),
        subject=subj,
        topic=topic,
    )


def sample_humaneval(rng: random.Random, difficulty: str) -> BenchmarkSeed:
    ds = _get_cached("openai_humaneval_test", _load_humaneval_test)
    idx, row = _pick_row(ds, rng)
    prompt = str(row.get("prompt", ""))
    canon = str(row.get("canonical_solution") or row.get("completion") or "")
    prob = _truncate(prompt, MAX_PROBLEM_CHARS)
    ref = _truncate(canon, MAX_REFERENCE_CHARS)
    subj, topic = _coding_subject_topic(difficulty, rng)
    tid = row.get("task_id", idx)
    return BenchmarkSeed(
        problem_statement=prob,
        reference_solution=ref,
        seed_dataset="openai/openai_humaneval/test",
        seed_example_id=str(tid),
        subject=subj,
        topic=topic,
    )


def sample_undergrad_coding_fallback(rng: random.Random, difficulty: str) -> BenchmarkSeed:
    """Undergraduate coding tier: MBPP or HumanEval only.

    ``codeparrot/apps`` (APPS) uses Hub dataset scripts incompatible with modern ``datasets``.
    """
    return sample_humaneval(rng, difficulty) if rng.random() < 0.5 else sample_mbpp(rng, difficulty)


def sample_benchmark_seed(rng: random.Random, difficulty: str, task_domain: str) -> BenchmarkSeed:
    """Sample one benchmark row for the given pipeline difficulty and domain."""
    if task_domain == "math":
        if difficulty == "high_school_low":
            return sample_svamp(rng, difficulty) if rng.random() < 0.5 else sample_gsm8k(rng, difficulty)
        if difficulty == "high_school_high":
            return sample_math_train_row(rng, difficulty)
        if difficulty == "undergraduate":
            return sample_math_undergrad(rng, difficulty)
        raise ValueError(f"unknown difficulty for math: {difficulty}")
    if task_domain == "coding":
        if difficulty == "high_school_low":
            return sample_mbpp(rng, difficulty)
        if difficulty == "high_school_high":
            return sample_humaneval(rng, difficulty)
        if difficulty == "undergraduate":
            return sample_undergrad_coding_fallback(rng, difficulty)
        raise ValueError(f"unknown difficulty for coding: {difficulty}")
    raise ValueError(f"unknown task_domain: {task_domain}")

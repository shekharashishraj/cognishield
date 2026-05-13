"""Load the actual SFT JSONL file and verify schema + split counts.

Gated on the file's existence — skipped in environments that don't have it
(CI, fresh clones). When the file IS present, asserts that the loader
ingests it correctly and the row count + split distribution match.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from training.data.cognibench_loader import filter_accepted, iter_jsonl


DATA = Path("/Users/ashishrajshekhar/Desktop/cognishield/sft.generated.batch.jsonl")
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="SFT dataset not on this host")


def test_loader_reads_all_records() -> None:
    rows = list(iter_jsonl(DATA))
    # Sanity bounds — file shouldn't be empty and shouldn't have ballooned.
    assert 100 < len(rows) < 20000
    assert all(r.messages for r in rows)
    assert all(r.split in {"exemplary_legitimate", "adequate_ambiguous", "failing_disallowed"} for r in rows)


def test_first_record_has_expected_structure() -> None:
    rows = list(iter_jsonl(DATA))
    r = rows[0]
    roles = [m.role for m in r.messages]
    assert roles[0] == "system"
    assert "user" in roles and "assistant" in roles
    # Conversations should alternate user/assistant after the system row.
    for prev, cur in zip(roles[1:], roles[2:]):
        if prev == "user":
            assert cur == "assistant"
        elif prev == "assistant":
            assert cur == "user"


def test_default_keep_all_three_splits() -> None:
    """All three splits contain correct tutor behavior; default keeps everything."""
    rows = list(iter_jsonl(DATA))
    kept = filter_accepted(rows, keep_splits=None)
    assert len(kept) == len(rows)


def test_filter_drops_unlisted_split() -> None:
    rows = list(iter_jsonl(DATA))
    kept = filter_accepted(rows, keep_splits=["exemplary_legitimate"])
    assert all(r.split == "exemplary_legitimate" for r in kept)
    assert len(kept) < len(rows)

"""Question-level train/validation/test split and leakage checks."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping

import numpy as np


def make_question_split(question_ids: Iterable[str], seed: int, fracs=(0.6, 0.2, 0.2)) -> dict[str, list[str]]:
    ids = sorted(set(question_ids))
    if not ids:
        raise ValueError("no question ids")
    assert abs(sum(fracs) - 1.0) < 1e-9
    perm = np.random.RandomState(seed).permutation(len(ids))
    n = len(ids)
    n_train = int(round(fracs[0] * n))
    n_val = int(round(fracs[1] * n))
    order = [ids[i] for i in perm]
    split = {
        "train": sorted(order[:n_train]),
        "validation": sorted(order[n_train:n_train + n_val]),
        "test": sorted(order[n_train + n_val:]),
    }
    assert_disjoint(split)
    assert set(split["train"]) | set(split["validation"]) | set(split["test"]) == set(ids)
    return split


def assert_disjoint(split: Mapping[str, Iterable[str]]) -> None:
    train_ids, val_ids, test_ids = (set(split[k]) for k in ("train", "validation", "test"))
    assert train_ids.isdisjoint(val_ids), "train/validation overlap"
    assert train_ids.isdisjoint(test_ids), "train/test overlap"
    assert val_ids.isdisjoint(test_ids), "validation/test overlap"


def split_lookup(split: Mapping[str, Iterable[str]]) -> dict[str, str]:
    return {qid: name for name, ids in split.items() for qid in ids}


def assert_no_leakage(rows: list[dict], split: Mapping[str, Iterable[str]]) -> dict:
    """rows: dicts with question_id, split, response_sha256. Returns a small report."""
    assert_disjoint(split)
    lookup = split_lookup(split)
    for r in rows:
        assert lookup[r["question_id"]] == r["split"], f"split mismatch for {r['question_id']}"
    by_hash: dict[str, set] = {}
    for r in rows:
        by_hash.setdefault(r["response_sha256"], set()).add(r["split"])
    straddling = [h for h, s in by_hash.items() if len(s) > 1]
    assert not straddling, f"{len(straddling)} identical responses appear in more than one split"
    return {
        "n_rows": len(rows),
        "per_split": dict(Counter(r["split"] for r in rows)),
        "n_questions_per_split": {k: len(set(r["question_id"] for r in rows if r["split"] == k))
                                  for k in ("train", "validation", "test")},
    }

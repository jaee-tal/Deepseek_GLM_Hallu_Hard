"""Shared pooled-data loading, weighting, and sampling for transfer probes."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import Config, split_path
from .transfer_experiments import load_run_data, organic_weights, source_view, source_weights
from .vendor.io_utils import read_json, read_jsonl

DATASETS = ("pop_qa", "trivia_qa", "hallu_hard")
REGISTERS = ("lie", "misremember", "unknown")
PROBES = ("dim", "logistic", "lda", "ridge")
BUDGETS = (.05, .10, .20)
MASSES = (.5, .8)
SEEDS = (42, 43, 44, 45, 46)


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def duplicate_exclusions(cfg: Config) -> tuple[dict[str, set[str]], list[dict]]:
    """Exclude complete question groups when identical text crosses split boundaries."""
    by_text: dict[str, list[dict]] = {}
    for dataset in DATASETS:
        split = read_json(split_path(cfg, dataset))
        lookup = {qid: name for name in ("train", "validation", "test") for qid in split[name]}
        for row in read_jsonl(cfg.data_dir(dataset) / "questions.jsonl"):
            key = " ".join(row["question"].casefold().split())
            by_text.setdefault(key, []).append({"dataset": dataset, "question_id": row["question_id"],
                                                 "split": lookup[row["question_id"]], "question": row["question"]})
    collisions = [group for group in by_text.values()
                  if len({item["dataset"] for item in group}) > 1
                  and len({item["split"] for item in group}) > 1]
    excluded = {dataset: set() for dataset in DATASETS}
    for group in collisions:
        for row in group:
            excluded[row["dataset"]].add(row["question_id"])
    return excluded, [row for group in collisions for row in group]


@dataclass
class PooledData:
    cfg: Config
    runs: dict
    source: dict[str, pd.DataFrame]
    organic: pd.DataFrame
    excluded: dict[str, set[str]]
    collisions: list[dict]
    model: str = "qwen3_8b"

    @classmethod
    def load(cls, cfg: Config, tag: str = "pilot", model: str = "qwen3_8b") -> "PooledData":
        excluded, collisions = duplicate_exclusions(cfg)
        runs = {dataset: load_run_data(cfg, dataset, tag=tag, model=model) for dataset in DATASETS}
        source = {}
        for register in REGISTERS:
            parts = []
            for dataset in DATASETS:
                part = source_view(runs[dataset].source_rows, register, "crossed")
                part = part.loc[~part["question_id"].isin(excluded[dataset])].copy()
                part["dataset"] = dataset
                parts.append(part)
            source[register] = pd.concat(parts, ignore_index=True)
        organic_parts = []
        for dataset in DATASETS:
            part = runs[dataset].organic_rows
            part = part.loc[~part["question_id"].isin(excluded[dataset])].copy()
            part["dataset"] = dataset
            organic_parts.append(part)
        organic = pd.concat(organic_parts, ignore_index=True)
        # The run loader verifies within-dataset question disjointness. Dataset IDs
        # are prefixed; exact cross-dataset text collisions are excluded above.
        for register in REGISTERS:
            for split in ("train", "validation", "test"):
                part = source[register].loc[source[register]["split"] == split]
                if part.groupby("dataset")["label"].nunique().to_dict() != {d: 2 for d in DATASETS}:
                    raise ValueError(f"missing source class: {register}/{split}")
        return cls(cfg, runs, source, organic, excluded, collisions, model)

    def source_for(self, register: str, variant: str = "crossed") -> pd.DataFrame:
        # ``source`` stores the superset used by the crossed design. Reapply the
        # view here so the basic baseline reuses the exact same cached rows.
        return source_view(self.source[register], register, variant)

    def X(self, rows: pd.DataFrame, layer: int, domain: str) -> np.ndarray:
        """Preserve the dataset-block row order of pooled views."""
        blocks = []
        for dataset in DATASETS:
            part = rows.loc[rows["dataset"] == dataset]
            if len(part):
                fn = self.runs[dataset].source_X if domain == "source" else self.runs[dataset].organic_X
                blocks.append(fn(part, layer))
        actual = rows["dataset"].tolist()
        expected = [d for d in DATASETS for _ in range(actual.count(d))]
        if actual != expected:
            raise AssertionError("pooled activation rows lost dataset-block ordering")
        return np.concatenate(blocks, axis=0)

    def signature(self, register: str, variant: str = "crossed") -> str:
        rows = self.source_for(register, variant)
        organic = self.organic
        value = {"register": register,
                 "source": rows[["conversation_id", "label", "split"]].to_dict("list"),
                 "organic": organic[["conversation_id", "label", "split"]].to_dict("list"),
                 "cache": {d: read_json(self.runs[d].run_dir / "results" / "transfer_experiments" /
                                         "cache" / "manifest.json") for d in DATASETS},
                 "excluded": {d: sorted(self.excluded[d]) for d in DATASETS}}
        # Preserve the exact historical crossed signature so its fitted models
        # are reused byte-for-byte. Basic gets its own distinct signature.
        if variant != "crossed":
            value["variant"] = variant
        return fingerprint(value)


def equal_dataset_weights(rows: pd.DataFrame, domain: str, variant: str = "crossed") -> np.ndarray:
    rows = rows.reset_index(drop=True)
    w = np.empty(len(rows), dtype=np.float64)
    for dataset in DATASETS:
        idx = np.flatnonzero(rows["dataset"].to_numpy() == dataset)
        part = rows.iloc[idx]
        if not len(part):
            raise ValueError(f"missing {dataset} {domain} rows")
        local = source_weights(part, variant) if domain == "source" else organic_weights(part)
        w[idx] = local / len(DATASETS)
    assert np.all(w > 0) and np.isclose(w.sum(), 1)
    return w


def dataset_macro_auc(rows: pd.DataFrame, scores: np.ndarray) -> float:
    if len(rows) != len(scores):
        raise ValueError("score length mismatch")
    values = []
    for dataset in DATASETS:
        mask = rows["dataset"].to_numpy() == dataset
        y = rows.loc[mask, "label"].to_numpy()
        if len(np.unique(y)) != 2:
            raise ValueError(f"single-class validation for {dataset}")
        values.append(roc_auc_score(y, scores[mask]))
    return float(np.mean(values))


def macro_ba_threshold(rows: pd.DataFrame, scores: np.ndarray) -> float:
    """One threshold maximizing equal-dataset mean validation balanced accuracy."""
    scores = np.asarray(scores, dtype=np.float64)
    candidates = np.r_[np.nextafter(np.max(scores), np.inf), np.unique(scores)[::-1]]
    values = np.zeros(len(candidates), dtype=np.float64)
    for dataset in DATASETS:
        sub = rows.loc[rows["dataset"] == dataset]
        s = scores[rows["dataset"].to_numpy() == dataset]
        y = sub["label"].to_numpy()
        pos, neg = np.sort(s[y == 1]), np.sort(s[y == 0])
        if not len(pos) or not len(neg):
            raise ValueError(f"single-class threshold validation for {dataset}")
        tpr = (len(pos) - np.searchsorted(pos, candidates, side="left")) / len(pos)
        fpr = (len(neg) - np.searchsorted(neg, candidates, side="left")) / len(neg)
        values += (1 + tpr - fpr) / (2 * len(DATASETS))
    return float(candidates[np.argmax(values)])


def organic_question_sample(rows: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Nested seed-specific question draws, stratified within each dataset."""
    train = rows.loc[rows["split"] == "train"]
    chosen = set()
    for dataset in DATASETS:
        sub = train.loc[train["dataset"] == dataset]
        qclass = sub.groupby("question_id")["label"].max()
        for label in (0, 1):
            qids = np.array(sorted(qclass.index[qclass == label]))
            rng = np.random.RandomState(seed + 1009 * DATASETS.index(dataset) + 37 * label)
            count = max(1, int(round(fraction * len(qids))))
            chosen.update(rng.permutation(qids)[:count])
    sampled = train.loc[train["question_id"].isin(chosen)].copy()
    equal_dataset_weights(sampled, "organic")
    return sampled

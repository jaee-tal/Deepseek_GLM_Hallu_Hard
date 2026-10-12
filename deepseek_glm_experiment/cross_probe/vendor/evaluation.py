"""Metrics, validation-chosen thresholds, question-level paired bootstrap, matched subset, permutation."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score


def choose_threshold(scores: np.ndarray, y: np.ndarray, rule: str = "balanced_accuracy") -> float:
    """Scan midpoints between sorted unique validation scores; maximise balanced accuracy (= Youden's J) or F1."""
    s = np.unique(scores)
    if len(s) == 1:
        return float(s[0])
    cands = np.concatenate([[s[0] - 1e-6], (s[:-1] + s[1:]) / 2, [s[-1] + 1e-6]])
    best_t, best_v = cands[0], -1.0
    for t in cands:
        pred = (scores >= t).astype(int)
        v = balanced_accuracy_score(y, pred) if rule == "balanced_accuracy" else f1_score(y, pred, zero_division=0)
        if v > best_v:
            best_v, best_t = v, t
    return float(best_t)


def metrics(scores: np.ndarray, y: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (scores >= threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(y, scores)),
        "auprc": float(average_precision_score(y, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "prevalence": float(y.mean()),
        "n": int(len(y)),
        "n_pos": int(y.sum()),
    }


def bootstrap_indices(qids: np.ndarray, n_boot: int, seed: int) -> list[np.ndarray]:
    """Resample question ids with replacement; return row-index arrays (with multiplicity). Shared across probes."""
    uq = np.unique(qids)
    rows_by_q = {q: np.flatnonzero(qids == q) for q in uq}
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uq, size=len(uq), replace=True)
        out.append(np.concatenate([rows_by_q[q] for q in pick]))
    return out


def bootstrap_metrics(scores: np.ndarray, y: np.ndarray, threshold: float, boots: list[np.ndarray]) -> dict[str, Any]:
    vals: dict[str, list[float]] = {"auroc": [], "auprc": [], "balanced_accuracy": [], "f1": []}
    skipped = 0
    for idx in boots:
        yy = y[idx]
        if yy.min() == yy.max():
            skipped += 1
            continue
        m = metrics(scores[idx], yy, threshold)
        for k in vals:
            vals[k].append(m[k])
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [float("nan")] * 2 for k, v in vals.items()}
    return {"ci95": ci, "n_boot_used": len(vals["auroc"]), "n_boot_skipped_single_class": skipped,
            "auroc_samples": np.asarray(vals["auroc"], dtype=np.float32)}


def paired_delta_ci(a_samples: np.ndarray, b_samples: np.ndarray) -> list[float] | None:
    """CI of (a - b) when both were computed on identical bootstrap resamples (same skipped set assumed)."""
    if len(a_samples) != len(b_samples) or len(a_samples) == 0:
        return None
    d = a_samples - b_samples
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def matched_subset(qids: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Row mask for questions with at least one label-0 and one label-1 response."""
    mask = np.zeros(len(y), dtype=bool)
    for q in np.unique(qids):
        rows = qids == q
        if y[rows].min() == 0 and y[rows].max() == 1:
            mask[rows] = True
    return mask


def within_question_auroc(scores: np.ndarray, y: np.ndarray, qids: np.ndarray) -> float | None:
    vals = []
    for q in np.unique(qids):
        rows = qids == q
        if y[rows].min() == 0 and y[rows].max() == 1:
            vals.append(roc_auc_score(y[rows], scores[rows]))
    return float(np.mean(vals)) if vals else None


def question_block_permutation(y: np.ndarray, qids: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Permute labels by shuffling whole question blocks' label sets across questions of equal size when possible;
    falls back to a plain row permutation. Both variants are reported by the caller."""
    uq = np.unique(qids)
    blocks = {q: y[qids == q].copy() for q in uq}
    by_size: dict[int, list] = {}
    for q, b in blocks.items():
        by_size.setdefault(len(b), []).append(q)
    out = y.copy()
    for size, qs in by_size.items():
        perm = rng.permutation(len(qs))
        for q, j in zip(qs, perm):
            out[qids == q] = rng.permutation(blocks[qs[j]])
    return out

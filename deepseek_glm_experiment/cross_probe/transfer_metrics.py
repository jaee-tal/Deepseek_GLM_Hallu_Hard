"""Metrics shared by the Qwen and GPT-OSS transfer reports."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

DATASETS = ("pop_qa", "trivia_qa", "hallu_hard")
METRICS = ("auroc", "auprc", "precision", "recall", "fpr", "fnr", "balanced_accuracy")
PROBE_LABELS = {
    "dim": "Difference of means",
    "logistic": "Logistic regression",
    "lda": "LDA",
    "ridge": "Ridge classifier",
}


def full_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    """Binary metrics with factual error/hallucination as the positive class."""
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    if len(y) != len(score) or set(np.unique(y)) != {0, 1}:
        raise ValueError("metrics require aligned scores and both labels")
    pred = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    truth_precision = tn / (tn + fn) if tn + fn else 0.0
    truth_recall = tn / (tn + fp)
    return {
        "n": int(len(y)), "truth": int((y == 0).sum()), "hallucination": int((y == 1).sum()),
        "tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp),
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "precision": float(precision), "recall": float(recall),
        "fpr": float(fpr), "fnr": float(fnr),
        "balanced_accuracy": float((recall + 1.0 - fpr) / 2.0),
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "truth_precision": float(truth_precision), "truth_recall": float(truth_recall),
        "hallucination_precision": float(precision), "hallucination_recall": float(recall),
    }


def macro_table(records: pd.DataFrame) -> pd.DataFrame:
    """Average seeds within datasets, then weight the three datasets equally."""
    rates = list(METRICS) + ["f1", "truth_precision", "truth_recall",
                              "hallucination_precision", "hallucination_recall"]
    first = records.groupby(["probe", "dataset", "register", "method"], as_index=False)[rates].mean()
    coverage = first.groupby(["probe", "register", "method"]).size()
    if not (coverage == len(DATASETS)).all():
        raise ValueError("incomplete dataset coverage")
    return first.groupby(["probe", "register", "method"], as_index=False)[rates].mean()


def pooled_confusions(records: pd.DataFrame) -> pd.DataFrame:
    """Pool datasets after averaging repeated fits within each dataset."""
    first = records.groupby(["probe", "dataset", "register", "method"], as_index=False)[
        ["tn", "fp", "fn", "tp"]].mean()
    return first.groupby(["probe", "register", "method"], as_index=False)[
        ["tn", "fp", "fn", "tp"]].sum()

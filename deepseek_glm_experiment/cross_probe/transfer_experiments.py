"""Outcome-controlled induced probes and organic-label adaptation.

This module deliberately reads the original replay and judge artifacts but writes only under
``results/transfer_experiments``.  The published cross-probing cells are never refitted or changed.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.linalg
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from .config import Config, split_path
from .dataset import backfill_framing, write_layer_major
from .prompts import TRUTHFUL_FOR
from .vendor.io_utils import read_json, read_jsonl, write_json
from .vendor.splits import split_lookup

log = logging.getLogger(__name__)
ACTIVE_FRAMINGS = ("lie", "misremember", "unknown")
PROBES = ("logistic", "lda", "dim", "ridge")
VARIANTS = ("basic", "crossed")
SPLITS = ("train", "validation", "test")
ANALYSIS_VERSION = 3


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def assemble_instructed(index: pd.DataFrame, verdicts: pd.DataFrame, split: dict[str, Any], known: set[str]) -> pd.DataFrame:
    """Recover both outcome labels, including instruction failures excluded by the original I0/I1 design."""
    assert not index["conversation_id"].duplicated().any(), "replay index contains repeated conversations"
    verdicts = verdicts.drop_duplicates("conversation_id", keep="last")[["conversation_id", "verdict"]]
    d = index.merge(verdicts, on="conversation_id", how="left", validate="one_to_one")
    d["framing"] = backfill_framing(d)
    lookup = split_lookup({s: split[s] for s in SPLITS})
    d["split"] = d["question_id"].map(lookup)
    mask = (d["condition"].isin(("truthful", "incorrect"))
            & d["question_id"].isin(known)
            & d["split"].isin(SPLITS)
            & d["alignment_ok"].astype(bool)
            & ~d["is_duplicate"].astype(bool)
            & (d["exclude_reason"].isna() | d["exclude_reason"].eq(""))
            & d["verdict"].isin(("correct", "incorrect")))
    d = d.loc[mask].copy()
    d = d[((d["condition"] == "truthful") & d["framing"].isin(("v1", "strict")))
          | ((d["condition"] == "incorrect") & d["framing"].isin(ACTIVE_FRAMINGS))].copy()
    d["label"] = (d["verdict"] == "incorrect").astype(np.int8)
    d["instruction_label"] = (d["condition"] == "incorrect").astype(np.int8)
    # Test questions have both a seen and a held-out paraphrase (template 4).
    d["held_out_template"] = d["template_id"].eq(4)
    d = d.sort_values("row").reset_index(drop=True)
    d["cache_row"] = np.arange(len(d), dtype=np.int64)
    assert not d["conversation_id"].duplicated().any()
    return d[["row", "cache_row", "question_id", "conversation_id", "split", "condition", "framing",
              "template_id", "held_out_template", "verdict", "label", "instruction_label"]]


def source_view(rows: pd.DataFrame, framing: str, variant: str) -> pd.DataFrame:
    if framing not in ACTIVE_FRAMINGS or variant not in VARIANTS:
        raise ValueError((framing, variant))
    err = (rows["condition"] == "incorrect") & (rows["framing"] == framing)
    truth = (rows["condition"] == "truthful") & (rows["framing"] == TRUTHFUL_FOR[framing])
    if variant == "basic":
        # Original experiment: factual-instruction truths versus successful
        # error-inducing hallucinations. This is intentionally instruction-
        # confounded and serves as the plain inorganic-transfer baseline.
        return rows.loc[(truth & rows["label"].eq(0)) | (err & rows["label"].eq(1))].copy()
    # Outcome-controlled/same-instruction experiment: both judged outcomes
    # occur within both instruction conditions, with equal cell weight.
    return rows.loc[err | truth].copy()


def _equal_cell_question_weights(rows: pd.DataFrame, cells: list[str]) -> np.ndarray:
    """Each cell has equal mass; within a cell each question has equal mass."""
    if rows.empty:
        raise ValueError("empty training view")
    w = np.zeros(len(rows), dtype=np.float64)
    groups = rows.reset_index(drop=True).groupby(cells, sort=False, dropna=False).indices
    for idx in groups.values():
        sub = rows.iloc[idx]
        counts = sub.groupby("question_id")["question_id"].transform("size").to_numpy()
        w[idx] = 1.0 / (len(groups) * sub["question_id"].nunique() * counts)
    assert np.all(w > 0) and np.isclose(w.sum(), 1.0)
    return w


def source_weights(rows: pd.DataFrame, variant: str) -> np.ndarray:
    if variant == "crossed":
        cells, expected = ["condition", "label"], 4
    elif variant == "basic":
        cells, expected = ["label"], 2
        observed = set(map(tuple, rows[["condition", "label"]].drop_duplicates().to_numpy()))
        if observed != {("truthful", 0), ("incorrect", 1)}:
            raise ValueError("basic requires truthful truths and error-instruction hallucinations")
    else:
        raise ValueError(f"unknown source design {variant!r}")
    if rows.groupby(cells, dropna=False).ngroups != expected:
        raise ValueError(f"{variant} requires {expected} nonempty instruction/outcome cells")
    return _equal_cell_question_weights(rows, cells)


def organic_weights(rows: pd.DataFrame) -> np.ndarray:
    if rows["label"].nunique() != 2:
        raise ValueError("organic training subset has a single outcome")
    return _equal_cell_question_weights(rows, ["label"])


def assert_question_split(source: pd.DataFrame, organic: pd.DataFrame) -> None:
    for d in (source, organic):
        q = {s: set(d.loc[d["split"] == s, "question_id"]) for s in SPLITS}
        assert not (q["train"] & q["validation"] or q["train"] & q["test"] or q["validation"] & q["test"])
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        qa = set(source.loc[source["split"] == a, "question_id"]) | set(organic.loc[organic["split"] == a, "question_id"])
        qb = set(source.loc[source["split"] == b, "question_id"]) | set(organic.loc[organic["split"] == b, "question_id"])
        assert not qa & qb, f"question leakage between {a} and {b}"


@dataclass
class RunData:
    run_dir: Path
    source_rows: pd.DataFrame
    source_acts: np.ndarray  # [layers, eligible instructed rows, hidden]
    organic_rows: pd.DataFrame
    organic_acts: np.ndarray  # existing [layers, original usable rows, hidden]
    known: set[str]

    @property
    def layers(self) -> int:
        return int(self.source_acts.shape[0])

    def source_X(self, rows: pd.DataFrame, layer: int) -> np.ndarray:
        return np.asarray(self.source_acts[layer, rows["cache_row"].to_numpy()], dtype=np.float32)

    def organic_X(self, rows: pd.DataFrame, layer: int) -> np.ndarray:
        return np.asarray(self.organic_acts[layer, rows["ds_row"].to_numpy()], dtype=np.float32)


def load_run_data(cfg: Config, dataset: str, tag: str = "pilot", model: str = "qwen3_8b") -> RunData:
    run_dir = cfg.run_dir(tag, model, dataset)
    out = run_dir / "results" / "transfer_experiments" / "cache"
    out.mkdir(parents=True, exist_ok=True)
    index = pd.DataFrame(read_jsonl(run_dir / "replay" / "index.jsonl"))
    verdicts = pd.DataFrame(read_jsonl(run_dir / "judge" / "verdicts.jsonl"))
    known_path = run_dir / "known_questions.json"
    if not known_path.exists():
        known_path = cfg.data_dir(dataset) / f"known_{model}.json"
    known_meta = read_json(known_path)
    if cfg.model(model).get("strict_organic_eligibility") and known_meta.get("criterion") != "correct_never_wrong":
        raise ValueError(f"{model} requires strict organic eligibility: {known_path}")
    known = set(known_meta["question_ids"])
    rows = assemble_instructed(index, verdicts, read_json(split_path(cfg, dataset)), known)
    organic = pd.read_parquet(run_dir / "dataset" / "rows.parquet")
    if "ds_row" not in organic.columns:
        organic["ds_row"] = np.arange(len(organic))
    organic = organic.loc[organic["condition"] == "organic"].copy()
    assert_question_split(rows, organic)
    manifest_path, acts_path = out / "manifest.json", out / "acts_layer_major.npy"
    prior = read_json(manifest_path) if manifest_path.exists() else {}
    row_fp = _hash({"conversations": rows["conversation_id"].tolist(), "rows": rows["row"].tolist()})
    cache_valid = (acts_path.exists() and prior.get("n_rows") == len(rows)
                   and prior.get("row_fingerprint", row_fp) == row_fp)
    replay_path = run_dir / "replay" / "activations.npy"
    if not cache_valid:
        if not replay_path.exists():
            raise FileNotFoundError(f"missing both reusable source cache and raw replay activations: {run_dir}")
        log.info("building eligible instructed activation cache: %s (%d rows)", dataset, len(rows))
        tmp = out / "acts_layer_major.tmp.npy"
        write_layer_major(replay_path, rows["row"].to_numpy(), tmp, log)
        tmp.replace(acts_path)
    if prior.get("row_fingerprint") != row_fp:
        write_json(manifest_path, {"row_fingerprint": row_fp, "n_rows": len(rows),
                                   "shape": list(np.load(acts_path, mmap_mode="r").shape),
                                   "source": "replay/activations.npy (raw intermediate may be regenerated)"})
    source_acts = np.load(acts_path, mmap_mode="r")
    organic_acts = np.load(run_dir / "dataset" / "acts_layer_major.npy", mmap_mode="r")
    assert source_acts.shape[1] == len(rows)
    assert source_acts.shape[0] == organic_acts.shape[0]
    assert source_acts.shape[2] == organic_acts.shape[2]
    return RunData(run_dir, rows, source_acts, organic, organic_acts, known)


@dataclass
class LinearProbe:
    coef: np.ndarray
    intercept: float
    center: np.ndarray
    scale: np.ndarray
    name: str
    param: Any

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.center.size:
            X = (X - self.center) / self.scale
        return X @ self.coef + self.intercept

    def save(self, path: Path) -> None:
        np.savez(path, coef=self.coef.astype(np.float32), intercept=np.float64(self.intercept),
                 center=self.center.astype(np.float32), scale=self.scale.astype(np.float32),
                 name=self.name, param=json.dumps(self.param))

    @classmethod
    def load(cls, path: Path) -> "LinearProbe":
        d = np.load(path, allow_pickle=False)
        return cls(d["coef"].astype(np.float64), float(d["intercept"]), d["center"].astype(np.float64),
                   d["scale"].astype(np.float64), str(d["name"]), json.loads(str(d["param"])))


def _weighted_center_scale(X: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.average(X, axis=0, weights=w)
    variance = np.average((X - mu) ** 2, axis=0, weights=w)
    scale = np.sqrt(variance)
    scale[scale < 1e-12] = 1.0
    return (X - mu) / scale, mu, scale


def fit_linear(name: str, param: Any, X: np.ndarray, y: np.ndarray, weights: np.ndarray,
               params: dict[str, Any]) -> LinearProbe:
    """All probe families use the same class/domain/question weights; LDA uses fixed shrinkage.

    scikit-learn's Ledoit-Wolf estimator does not accept sample weights, so the weighted experiment
    searches the configured numeric LDA shrinkages and does not mislabel an approximation as 'auto'.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int8)
    weights = np.asarray(weights, dtype=np.float64)
    if y.min() == y.max() or np.any(weights <= 0):
        raise ValueError("fit needs both labels and positive weights")
    if name == "dim":
        if params.get("dim_standardize", False):
            Z, mu, scale = _weighted_center_scale(X, weights)
        else:
            Z, mu, scale = X, np.zeros(0), np.zeros(0)
        coef = np.average(Z[y == 1], axis=0, weights=weights[y == 1]) - np.average(Z[y == 0], axis=0, weights=weights[y == 0])
        return LinearProbe(coef, 0.0, mu, scale, name, None)
    Z, mu, scale = _weighted_center_scale(X, weights)
    if name == "logistic":
        clf = LogisticRegression(C=float(param), max_iter=int(params.get("lr_max_iter", 5000)), solver="lbfgs")
        clf.fit(Z, y, sample_weight=weights)
        return LinearProbe(np.asarray(clf.coef_).reshape(-1),
                           float(np.asarray(clf.intercept_).reshape(-1)[0]),
                           mu, scale, name, param)
    if name == "ridge":
        clf = RidgeClassifier(alpha=float(param), solver=str(params.get("ridge_solver", "lsqr")),
                              tol=float(params.get("ridge_tol", 1e-4)),
                              max_iter=params.get("ridge_max_iter"))
        clf.fit(Z, y, sample_weight=weights)
        return LinearProbe(np.asarray(clf.coef_).reshape(-1),
                           float(np.asarray(clf.intercept_).reshape(-1)[0]),
                           mu, scale, name, param)
    if name == "lda":
        shrinkage = float(param)
        means, covs = [], []
        for c in (0, 1):
            z, wc = Z[y == c], weights[y == c]
            mean = np.average(z, axis=0, weights=wc)
            centered = z - mean
            cov = (centered.T * wc) @ centered / wc.sum()
            target = np.trace(cov) / cov.shape[0]
            cov *= 1.0 - shrinkage
            cov.flat[::cov.shape[0] + 1] += shrinkage * target
            means.append(mean); covs.append(cov)
        pooled = 0.5 * (covs[0] + covs[1])
        delta = means[1] - means[0]
        coef = scipy.linalg.solve(pooled, delta, assume_a="pos", check_finite=False)
        intercept = -0.5 * float((means[1] + means[0]) @ coef)
        return LinearProbe(coef, intercept, mu, scale, name, param)
    raise ValueError(name)


def grid_params(name: str, params: dict[str, Any]) -> list[Any]:
    if name == "dim":
        return [None]
    if name == "lda":
        return [float(x) for x in params["lda_shrinkage_grid"] if x != "auto"]
    if name == "logistic":
        return [float(x) for x in params["C_grid"]]
    if name == "ridge":
        return [float(x) for x in params.get(
            "ridge_alpha_grid", (.01, .1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0))]
    raise ValueError(name)


def fit_linear_candidates(name: str, candidates: list[Any], X: np.ndarray, y: np.ndarray,
                          weights: np.ndarray, params: dict[str, Any]):
    """Fit an outcome-selection grid, sharing preprocessing and covariance work."""
    if name == "dim":
        yield None, fit_linear(name, None, X, y, weights, params)
        return
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int8)
    weights = np.asarray(weights, dtype=np.float64)
    if y.min() == y.max() or np.any(weights <= 0):
        raise ValueError("fit needs both labels and positive weights")
    Z, mu, scale = _weighted_center_scale(X, weights)
    if name == "logistic":
        clf = LogisticRegression(C=float(candidates[0]), max_iter=int(params.get("lr_max_iter", 5000)),
                                 solver="lbfgs", warm_start=True)
        for param in candidates:
            clf.C = float(param)
            clf.fit(Z, y, sample_weight=weights)
            yield param, LinearProbe(np.asarray(clf.coef_).reshape(-1).copy(),
                                     float(np.asarray(clf.intercept_).reshape(-1)[0]),
                                     mu, scale, name, param)
        return
    if name == "ridge":
        for param in candidates:
            clf = RidgeClassifier(alpha=float(param), solver=str(params.get("ridge_solver", "lsqr")),
                                  tol=float(params.get("ridge_tol", 1e-4)),
                                  max_iter=params.get("ridge_max_iter"))
            clf.fit(Z, y, sample_weight=weights)
            yield param, LinearProbe(np.asarray(clf.coef_).reshape(-1).copy(),
                                     float(np.asarray(clf.intercept_).reshape(-1)[0]),
                                     mu, scale, name, param)
        return
    if name == "lda":
        means, covs = [], []
        for c in (0, 1):
            z, wc = Z[y == c], weights[y == c]
            mean = np.average(z, axis=0, weights=wc)
            centered = z - mean
            cov = (centered.T * wc) @ centered / wc.sum()
            means.append(mean)
            covs.append(cov)
        pooled = 0.5 * (covs[0] + covs[1])
        target = np.trace(pooled) / pooled.shape[0]
        delta = means[1] - means[0]
        for param in candidates:
            shrinkage = float(param)
            regularized = (1.0 - shrinkage) * pooled
            regularized.flat[::regularized.shape[0] + 1] += shrinkage * target
            coef = scipy.linalg.solve(regularized, delta, assume_a="pos", check_finite=False)
            intercept = -0.5 * float((means[1] + means[0]) @ coef)
            yield param, LinearProbe(coef, intercept, mu, scale, name, param)
        return
    raise ValueError(name)


def macro_outcome_auc(rows: pd.DataFrame, scores: np.ndarray) -> float:
    """Mean of the factual-outcome AUCs within each instruction condition."""
    values = []
    for _, sub in rows.reset_index(drop=True).groupby("condition"):
        y = sub["label"].to_numpy()
        if y.min() == y.max():
            raise ValueError("validation condition has only one outcome")
        values.append(roc_auc_score(y, scores[sub.index.to_numpy()]))
    return float(np.mean(values))


def source_validation_auc(rows: pd.DataFrame, scores: np.ndarray, variant: str) -> float:
    """Use the estimand appropriate to each source construction."""
    if variant == "basic":
        return float(roc_auc_score(rows["label"].to_numpy(), scores))
    return macro_outcome_auc(rows, scores)


def ranking_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float | int | None]:
    y = np.asarray(y, dtype=int); score = np.asarray(score, dtype=float)
    result: dict[str, float | int | None] = {"n": len(y), "n_pos": int(y.sum()), "auroc": None,
                                           "pauc_fpr_05": None, "pauc_fpr_10": None,
                                           "fnr_at_fpr_05": None, "fnr_at_fpr_10": None}
    if not len(y) or y.min() == y.max():
        return result
    result["auroc"] = float(roc_auc_score(y, score))
    # Preserve every empirical operating point: dropping collinear points can change
    # the best attainable TPR immediately below a strict FPR cap.
    fpr, tpr, _ = roc_curve(y, score, drop_intermediate=False)
    for cap, suffix in ((0.05, "05"), (0.10, "10")):
        result[f"pauc_fpr_{suffix}"] = float(roc_auc_score(y, score, max_fpr=cap))
        result[f"fnr_at_fpr_{suffix}"] = float(1.0 - np.max(tpr[fpr <= cap + 1e-12]))
    return result


def threshold_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float | int]:
    y = np.asarray(y, dtype=int); pred = np.asarray(score) >= threshold
    return {"n": len(y), "n_pos": int(y.sum()),
            "fpr": float(pred[y == 0].mean()) if np.any(y == 0) else float("nan"),
            "fnr": float((~pred[y == 1]).mean()) if np.any(y == 1) else float("nan")}


def balanced_accuracy_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Select Youden's J on validation only, in O(n log n) rather than a threshold scan."""
    fpr, tpr, thresholds = roc_curve(y, score)
    # The first argmax prefers the highest threshold on ties.
    return float(thresholds[np.argmax(tpr - fpr)])


def cluster_auc_ci(y: np.ndarray, score: np.ndarray, qids: np.ndarray, repeats: int, seed: int) -> list[float]:
    """Question bootstrap for the main AUROC; operating-point CIs are computed in the report stage."""
    rng = np.random.RandomState(seed)
    unique, inverse = np.unique(qids, return_inverse=True)
    by_q = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    values = []
    for _ in range(repeats):
        pick = rng.randint(len(unique), size=len(unique))
        idx = np.concatenate([by_q[j] for j in pick])
        if y[idx].min() != y[idx].max():
            values.append(roc_auc_score(y[idx], score[idx]))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))] if values else [float("nan")] * 2


def _data_fp(source: pd.DataFrame, organic: pd.DataFrame) -> str:
    return _hash({"source": source[["conversation_id", "split", "label", "cache_row"]].to_dict("records"),
                  "organic": organic[["conversation_id", "split", "label", "ds_row"]].to_dict("records")})


def fit_outcome(data: RunData, cfg: Config, framing: str, variant: str, probe: str,
                layers: list[int] | None = None) -> dict[str, Any]:
    source = source_view(data.source_rows, framing, variant)
    organic = data.organic_rows
    assert_question_split(source, organic)
    layers = list(range(data.layers)) if layers is None else sorted(set(layers))
    if not layers or any(layer < 0 or layer >= data.layers for layer in layers):
        raise ValueError(f"layers must be between 0 and {data.layers - 1}")
    params = cfg["probe_params"]
    candidates = grid_params(probe, params)
    out = data.run_dir / "results" / "transfer_experiments" / "outcome" / framing / variant / probe
    out.mkdir(parents=True, exist_ok=True)
    fp = _hash({"data": _data_fp(source, organic), "params": params, "candidates": candidates,
                "model": cfg.model(data.run_dir.parent.parent.name)["checkpoint"]})
    grid_path = out / "grid.json"
    grid = read_json(grid_path) if grid_path.exists() else {"fingerprint": fp, "layers": {}}
    if grid.get("fingerprint") != fp:
        log.info("source data or configuration changed; refitting the grid at %s", out)
        grid = {"fingerprint": fp, "layers": {}}
    tr = source.loc[source["split"] == "train"]
    va = source.loc[source["split"] == "validation"]
    te = source.loc[source["split"] == "test"]
    for part in (tr, va, te):
        source_weights(part, variant)  # early, specific error for missing cells
    w = source_weights(tr, variant) * len(tr)
    for layer in layers:
        model_path = out / f"layer_{layer:02d}_best.npz"
        if str(layer) in grid["layers"] and model_path.exists():
            continue
        Xtr, Xva = data.source_X(tr, layer), data.source_X(va, layer)
        ytr = tr["label"].to_numpy()
        best: tuple[float, Any, LinearProbe] | None = None
        scores_by_param = {}
        for param, clf in fit_linear_candidates(probe, candidates, Xtr, ytr, w, params):
            auc = source_validation_auc(va, clf.score(Xva), variant)
            scores_by_param[str(param)] = auc
            if best is None or auc > best[0]:
                best = (auc, param, clf)
        assert best is not None
        best[2].save(model_path)
        grid["layers"][str(layer)] = {"best_val_macro_auc": best[0], "best_param": best[1], "val_macro_auc": scores_by_param}
        write_json(grid_path, grid)
        log.debug("%s/%s/%s layer %d val macro AUC %.3f", framing, variant, probe, layer, best[0])
    # Selection always uses every completed layer for this data/config.
    available_layers = sorted(int(layer) for layer in grid["layers"])
    selected_layer = max(available_layers, key=lambda l: (grid["layers"][str(l)]["best_val_macro_auc"], -l))
    selected = grid["layers"][str(selected_layer)]
    log.info("%s/%s/%s selected layer %d, param %s, val macro AUC %.3f from %d layers",
             framing, variant, probe, selected_layer, selected["best_param"],
             selected["best_val_macro_auc"], len(available_layers))
    model = LinearProbe.load(out / f"layer_{selected_layer:02d}_best.npz")
    yva, yte = va["label"].to_numpy(), te["label"].to_numpy()
    val_scores = model.score(data.source_X(va, selected_layer))
    source_scores = model.score(data.source_X(te, selected_layer))
    org_test = organic.loc[organic["split"] == "test"]
    yorg = org_test["label"].to_numpy()
    organic_scores = model.score(data.organic_X(org_test, selected_layer))
    threshold = balanced_accuracy_threshold(yva, val_scores)
    known_mask = org_test["question_id"].isin(data.known).to_numpy()
    result = {"dataset": data.run_dir.parent.name, "framing": framing, "variant": variant, "probe": probe,
              "fingerprint": fp, "layer": selected_layer, "param": selected["best_param"],
              "val_macro_auc": selected["best_val_macro_auc"],
              "source_test_macro_auc": source_validation_auc(te, source_scores, variant),
              "source_test": ranking_metrics(yte, source_scores), "organic_test": ranking_metrics(yorg, organic_scores),
              "organic_known_test": ranking_metrics(yorg[known_mask], organic_scores[known_mask]),
              "organic_unknown_test": ranking_metrics(yorg[~known_mask], organic_scores[~known_mask]),
              "source_val_threshold": threshold,
              "organic_at_source_threshold": threshold_metrics(yorg, organic_scores, threshold),
              "organic_known_at_source_threshold": threshold_metrics(yorg[known_mask], organic_scores[known_mask], threshold),
              "organic_unknown_at_source_threshold": threshold_metrics(yorg[~known_mask], organic_scores[~known_mask], threshold),
              "source_at_source_threshold": threshold_metrics(yte, source_scores, threshold),
              "organic_auroc_ci95": cluster_auc_ci(yorg, organic_scores, org_test["question_id"].to_numpy(),
                                                   int(cfg["confidence_intervals"]["samples"]),
                                                   int(cfg["confidence_intervals"]["seed"])),
              "source_cells": {f"{c}/{lab}": int(((source["condition"] == c) & (source["label"] == lab)).sum())
                               for c in source["condition"].unique() for lab in (0, 1)},
              "source_cells_by_split": {split: {f"{condition}/{label}": int(((source["split"] == split)
                                                                               & (source["condition"] == condition)
                                                                               & (source["label"] == label)).sum())
                                                 for condition in source["condition"].unique() for label in (0, 1)}
                                        for split in SPLITS},
              "instruction_auc_on_source_test": (float(roc_auc_score(te["instruction_label"], source_scores))
                                                 if te["instruction_label"].nunique() == 2 else None),
              "heldout_source_test": {key: ranking_metrics(te.loc[mask, "label"].to_numpy(), source_scores[mask])
                                      for key, mask in (("seen", ~te["held_out_template"].to_numpy()),
                                                        ("heldout", te["held_out_template"].to_numpy()))},
              "note": "FNR@FPR caps describes the organic test ROC frontier, not a deployable test-selected threshold."}
    np.savez(out / "selected_scores.npz", source_validation=val_scores.astype(np.float32),
             source_test=source_scores.astype(np.float32), organic_test=organic_scores.astype(np.float32))
    write_json(out / "selection.json", {"fingerprint": fp, "layers": available_layers, "layer": selected_layer, "param": selected["best_param"],
                                        "selection_rule": ("maximum validation AUROC" if variant == "basic" else
                                                           "maximum mean within-instruction validation AUROC")
                                                          + "; ties -> shallower layer"})
    write_json(out / "results.json", result)
    return result


def sampled_organic_questions(rows: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Nested question-level budgets for a given seed; never returns validation or test rows."""
    train = rows.loc[rows["split"] == "train"]
    qids = np.array(sorted(train["question_id"].unique()))
    rng = np.random.RandomState(seed)
    chosen = set(rng.permutation(qids)[:max(1, int(round(fraction * len(qids))))])
    return train.loc[train["question_id"].isin(chosen)].copy()


def adaptation_dir(run_dir: Path, framing: str, probe: str, source_variant: str) -> Path:
    """Keep historical crossed paths stable while giving basic mixes a separate cache."""
    root = run_dir / "results" / "transfer_experiments"
    if source_variant == "crossed":
        return root / "adaptation" / framing / probe
    return root / "adaptation_basic" / framing / probe


def fit_adaptation(data: RunData, cfg: Config, framing: str, probe: str,
                   fractions: tuple[float, ...] = (0.05, 0.10, 0.20),
                   organic_masses: tuple[float, ...] = (0.5, 0.8),
                   seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
                   source_variant: str = "crossed") -> list[dict[str, Any]]:
    """Fixed-layer learning curve; all organic fitting labels come only from original training questions."""
    if source_variant not in VARIANTS:
        raise ValueError(source_variant)
    selection_path = (data.run_dir / "results" / "transfer_experiments" / "outcome" /
                      framing / source_variant / probe / "selection.json")
    if not selection_path.exists():
        raise RuntimeError(f"run {source_variant} outcome transfer first: {selection_path}")
    selection = read_json(selection_path)
    source = source_view(data.source_rows, framing, source_variant)
    organic = data.organic_rows
    if selection["fingerprint"] != _hash({"data": _data_fp(source, organic), "params": cfg["probe_params"],
                                           "candidates": grid_params(probe, cfg["probe_params"]),
                                           "model": cfg.model(data.run_dir.parent.parent.name)["checkpoint"]}):
        raise RuntimeError(f"{source_variant} selection was produced from different data/config")
    layer, param = int(selection["layer"]), selection["param"]
    src_train = source.loc[source["split"] == "train"]
    org_test = organic.loc[organic["split"] == "test"]
    assert_question_split(source, organic)
    Xsrc = data.source_X(src_train, layer)
    ysrc = src_train["label"].to_numpy()
    wsrc = source_weights(src_train, source_variant)
    Xtest = data.organic_X(org_test, layer)
    ytest = org_test["label"].to_numpy()
    qtest = org_test["question_id"].to_numpy()
    known_test = org_test["question_id"].isin(data.known).to_numpy()

    def evaluate_score(score: np.ndarray) -> dict[str, Any]:
        return {"organic_test": ranking_metrics(ytest, score),
                "organic_known_test": ranking_metrics(ytest[known_test], score[known_test]),
                "organic_unknown_test": ranking_metrics(ytest[~known_test], score[~known_test]),
                "organic_auroc_ci95": cluster_auc_ci(ytest, score, qtest,
                                                     min(200, int(cfg["confidence_intervals"]["samples"])),
                                                     int(cfg["confidence_intervals"]["seed"]))}

    source_only_model = LinearProbe.load(selection_path.parent / f"layer_{layer:02d}_best.npz")
    source_only_score = source_only_model.score(Xtest)
    source_only_eval = evaluate_score(source_only_score)
    out = adaptation_dir(data.run_dir, framing, probe, source_variant)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        for fraction in fractions:
            chosen = sampled_organic_questions(organic, fraction, seed)
            if chosen["label"].nunique() != 2:
                raise ValueError(f"organic budget {fraction:g}, seed {seed} is single-class")
            Xorg = data.organic_X(chosen, layer)
            yorg = chosen["label"].to_numpy()
            worg = organic_weights(chosen)
            for mass in organic_masses:
                fp = _hash({"source_selection": selection, "organic_conversations": chosen["conversation_id"].tolist(),
                            "organic_labels": yorg.tolist(), "fraction": fraction, "mass": mass, "seed": seed})
                stem = f"seed{seed}_budget{fraction:g}_organic{mass:g}"
                result_path, model_path = out / f"{stem}.json", out / f"{stem}.npz"
                if result_path.exists() and model_path.exists():
                    prior = read_json(result_path)
                    if prior.get("fingerprint") == fp:
                        if prior.get("analysis_version") != ANALYSIS_VERSION:
                            refreshed_score = LinearProbe.load(model_path).score(Xtest)
                            prior["organic_test"] = ranking_metrics(ytest, refreshed_score)
                            prior["organic_known_test"] = ranking_metrics(ytest[known_test], refreshed_score[known_test])
                            prior["organic_unknown_test"] = ranking_metrics(ytest[~known_test], refreshed_score[~known_test])
                            prior["analysis_version"] = ANALYSIS_VERSION
                            prior["source_selection_fingerprint"] = _hash(selection)
                            write_json(result_path, prior)
                        results.append(prior)
                        continue
                    log.info("refreshing %s after source selection or labels changed", result_path.name)
                if mass == 0.0:
                    model, evaluation = source_only_model, source_only_eval
                else:
                    parts_X, parts_y, parts_w = [], [], []
                    if mass < 1.0:
                        parts_X.append(Xsrc); parts_y.append(ysrc); parts_w.append((1.0 - mass) * wsrc)
                    parts_X.append(Xorg); parts_y.append(yorg); parts_w.append(mass * worg)
                    X = np.concatenate(parts_X); y = np.concatenate(parts_y)
                    # Keep the same effective total weight across budgets/masses, so logistic's C is comparable.
                    weights = np.concatenate(parts_w) * len(src_train)
                    model = fit_linear(probe, param, X, y, weights, cfg["probe_params"])
                    score = model.score(Xtest)
                    evaluation = evaluate_score(score)
                result = {"dataset": data.run_dir.parent.name, "framing": framing, "probe": probe,
                          "analysis_version": ANALYSIS_VERSION,
                          "source_variant": source_variant, "source_selection_fingerprint": _hash(selection),
                          "seed": seed, "budget_fraction": fraction,
                          "organic_mass": mass, "n_organic_questions": int(chosen["question_id"].nunique()),
                          "n_organic_rows": len(chosen), "n_organic_pos": int(yorg.sum()),
                          "n_source_rows": (len(src_train) if mass < 1 else 0), "layer": layer, "param": param,
                          "fingerprint": fp, **evaluation}
                model.save(model_path)
                write_json(result_path, result)
                results.append(result)
                log.debug("adapt %s/%s seed %d budget %.0f%% organic mass %.0f%%: AUC %.3f",
                          framing, probe, seed, 100 * fraction, 100 * mass, result["organic_test"]["auroc"])
    # Keep the union of completed trials when a later command adds budgets or seeds.
    completed = [r for path in sorted(out.glob("seed*_budget*_organic*.json"))
                 if (r := read_json(path)).get("source_selection_fingerprint") == _hash(selection)]
    write_json(out / "results.json", completed)
    log.info("adapt %s/%s completed %d requested, %d current-selection trials", framing, probe, len(results), len(completed))
    return results


def summarize_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make flat, reviewable tables from completed cells without refitting any probes."""
    root = run_dir / "results" / "transfer_experiments"
    n_layers = int(np.load(run_dir / "dataset" / "acts_layer_major.npy", mmap_mode="r").shape[0])

    outcome_rows = []
    for path in sorted((root / "outcome").glob("*/*/*/results.json")):
        r = read_json(path)
        selection = read_json(path.parent / "selection.json")
        outcome_rows.append({"dataset": r["dataset"], "framing": r["framing"], "variant": r["variant"],
                             "probe": r["probe"], "layer": r["layer"], "param": r["param"],
                             "n_source_test": r["source_test"]["n"], "n_organic_test": r["organic_test"]["n"],
                             "source_test_macro_auc": r["source_test_macro_auc"],
                             "organic_auroc": r["organic_test"]["auroc"],
                             "organic_auroc_ci_low": r["organic_auroc_ci95"][0],
                             "organic_auroc_ci_high": r["organic_auroc_ci95"][1],
                             "organic_known_auroc": r["organic_known_test"]["auroc"],
                             "organic_fnr_at_fpr_05": r["organic_test"]["fnr_at_fpr_05"],
                             "organic_fnr_at_fpr_10": r["organic_test"]["fnr_at_fpr_10"],
                             "organic_fpr_source_threshold": r["organic_at_source_threshold"]["fpr"],
                             "organic_fnr_source_threshold": r["organic_at_source_threshold"]["fnr"],
                             "full_layer_grid": len(selection["layers"]) == n_layers,
                             "layer_grid": ",".join(map(str, selection["layers"]))})
    outcome = pd.DataFrame(outcome_rows)
    adaptation_rows = []
    paths = [(path, variant)
             for base, variant in (("adaptation", "crossed"), ("adaptation_basic", "basic"))
             for path in sorted((root / base).glob("*/*/results.json"))]
    for path, variant in paths:
        selected = read_json(root / "outcome" / path.parent.parent.name / variant /
                             path.parent.name / "selection.json")
        for r in read_json(path):
            adaptation_rows.append({"dataset": r["dataset"], "framing": r["framing"], "probe": r["probe"],
                                    "source_variant": r.get("source_variant", variant),
                                    "full_layer_grid": len(selected["layers"]) == n_layers,
                                    "seed": r["seed"], "budget_fraction": r["budget_fraction"],
                                    "organic_mass": r["organic_mass"],
                                    "organic_budget_questions": r["n_organic_questions"],
                                    "organic_budget_pos": r["n_organic_pos"],
                                    "n_organic_questions_used": (r["n_organic_questions"] if r["organic_mass"] > 0 else 0),
                                    "layer": r["layer"],
                                    "organic_auroc": r["organic_test"]["auroc"],
                                    "organic_fnr_at_fpr_05": r["organic_test"]["fnr_at_fpr_05"],
                                    "organic_fnr_at_fpr_10": r["organic_test"]["fnr_at_fpr_10"],
                                    "organic_auroc_ci_low": r["organic_auroc_ci95"][0],
                                    "organic_auroc_ci_high": r["organic_auroc_ci95"][1]})
    adaptation = pd.DataFrame(adaptation_rows)
    root.mkdir(parents=True, exist_ok=True)
    outcome.to_csv(root / "outcome_summary.csv", index=False)
    if not adaptation.empty:
        group = ["dataset", "framing", "probe", "source_variant", "budget_fraction", "organic_mass"]
        aggregate = adaptation.groupby(group, as_index=False).agg(
            n_seeds=("seed", "nunique"), mean_organic_auroc=("organic_auroc", "mean"),
            sd_organic_auroc=("organic_auroc", "std"),
            mean_fnr_at_fpr_05=("organic_fnr_at_fpr_05", "mean"),
            mean_fnr_at_fpr_10=("organic_fnr_at_fpr_10", "mean"))
        aggregate.to_csv(root / "adaptation_summary.csv", index=False)
    adaptation.to_csv(root / "adaptation_trials.csv", index=False)
    return outcome, adaptation

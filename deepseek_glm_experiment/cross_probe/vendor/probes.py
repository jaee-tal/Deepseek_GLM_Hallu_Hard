"""Linear probes with layer and regularization selection on validation data."""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import scipy.linalg
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import empirical_covariance, ledoit_wolf, shrunk_covariance
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

log = logging.getLogger(__name__)


def linear_form(clf) -> tuple[np.ndarray, float, np.ndarray | None, np.ndarray | None]:
    """Return (w, b, scaler_mean, scaler_scale) such that score(z) = w . ((z - mean)/scale) + b (mean/scale None => raw z)."""
    if isinstance(clf, DiffMeans):
        return np.asarray(clf.w), 0.0, clf.mu, clf.sd
    if isinstance(clf, Pipeline) and isinstance(clf.named_steps.get("clf"), ShrunkLDA):
        scaler, est = clf.named_steps["scaler"], clf.named_steps["clf"]
        return np.asarray(est.coef_).reshape(-1), float(est.intercept_), scaler.mean_.copy(), scaler.scale_.copy()
    scaler = clf.named_steps["scaler"]
    est = clf.named_steps["clf"]
    w = np.asarray(est.coef_).reshape(-1)
    b = float(np.asarray(est.intercept_).reshape(-1)[0])
    return w, b, scaler.mean_.copy(), scaler.scale_.copy()


def save_probe(clf, path_npz: Path, meta: dict[str, Any]) -> None:
    """Persist one fitted probe in a framework-independent form (+ meta) — called as soon as the fit exists."""
    w, b, mu, sd = linear_form(clf)
    path_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path_npz, w=w.astype(np.float32), b=np.float32(b),
             scaler_mean=(mu if mu is not None else np.zeros(0)).astype(np.float32),
             scaler_scale=(sd if sd is not None else np.zeros(0)).astype(np.float32),
             meta=json.dumps(meta, default=str))


def load_probe(path_npz: Path):
    """Return (score_fn, meta) for a probe saved with save_probe; score_fn maps [n, H] activations to scores."""
    d = np.load(path_npz, allow_pickle=False)
    w, b = d["w"].astype(np.float64), float(d["b"])
    mu, sd = d["scaler_mean"], d["scaler_scale"]
    meta = json.loads(str(d["meta"]))

    def score(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if mu.size:
            X = (X - mu) / sd
        return X @ w + b

    return score, meta


PROBE_NAMES = ["logistic", "svm", "lda", "ridge"]
SUPPLEMENTARY_PROBES = ["dim"]


class ShrunkLDA(ClassifierMixin, BaseEstimator):
    """Binary Fisher LDA with a shrunk pooled covariance: w = Sigma^-1 (mu_1 - mu_0).

    Covariance semantics are sklearn's (`_class_cov`: prior-weighted mean of the per-class shrunk covariances,
    Ledoit-Wolf for shrinkage="auto"), but the linear system is solved by Cholesky instead of sklearn's
    `scipy.linalg.lstsq`, which raised "SVD did not converge" on 4096-dimensional activation covariances and is
    roughly an order of magnitude slower. Priors are fixed at [0.5, 0.5], so the intercept carries no class-prior term.
    Falls back to a general solve, then to least squares, if the shrunk covariance is not positive definite.
    """

    def __init__(self, shrinkage: float | str = "auto"):
        self.shrinkage = shrinkage
        self.coef_ = None
        self.intercept_ = 0.0

    def _cov(self, Xg: np.ndarray) -> np.ndarray:
        """sklearn's `discriminant_analysis._cov`, verbatim: Ledoit-Wolf on per-feature-standardized data
        (then rescaled) for "auto", otherwise a fixed shrinkage of the empirical covariance."""
        if isinstance(self.shrinkage, str):
            if self.shrinkage != "auto":
                raise ValueError(self.shrinkage)
            sc = StandardScaler()
            Xs = sc.fit_transform(Xg)
            cov = ledoit_wolf(Xs)[0]
            return sc.scale_[:, None] * cov * sc.scale_[None, :]
        return shrunk_covariance(empirical_covariance(Xg), float(self.shrinkage))

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"ShrunkLDA is binary; got classes {classes}")
        self.classes_ = classes
        means = [X[y == c].mean(0) for c in classes]
        cov = 0.5 * self._cov(X[y == classes[0]]) + 0.5 * self._cov(X[y == classes[1]])
        b = np.stack(means, axis=1)                       # [d, 2]: solve once for both class means
        try:
            sol = scipy.linalg.solve(cov, b, assume_a="pos")
            self.solver_used_ = "cholesky"
        except Exception:                                  # noqa: BLE001 - not positive definite / ill conditioned
            try:
                sol = scipy.linalg.solve(cov, b)
                self.solver_used_ = "lu"
            except Exception:                              # noqa: BLE001
                sol = scipy.linalg.lstsq(cov, b, lapack_driver="gelsy")[0]
                self.solver_used_ = "lstsq_gelsy"
        w0, w1 = sol[:, 0], sol[:, 1]
        self.coef_ = w1 - w0
        self.intercept_ = float(-0.5 * (means[1] @ w1 - means[0] @ w0))
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[(self.decision_function(X) > 0).astype(int)]


class DiffMeans:
    """w = mean(X[y==1]) - mean(X[y==0]); score = X @ w (train statistics only)."""

    def __init__(self, standardize: bool = False):
        self.standardize = standardize
        self.w = None
        self.mu = None
        self.sd = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        if self.standardize:
            self.mu = X.mean(0)
            self.sd = X.std(0) + 1e-8
            X = (X - self.mu) / self.sd
        self.w = X[y == 1].mean(0) - X[y == 0].mean(0)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.standardize:
            X = (X - self.mu) / self.sd
        return X @ self.w


def make_probe(name: str, C: float | None, params: dict[str, Any]):
    cw = params.get("class_weight", "balanced")
    if name == "logistic":
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(penalty="l2", C=C, class_weight=cw, max_iter=int(params.get("lr_max_iter", 5000)), solver="lbfgs"))])
    if name == "ridge":
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", RidgeClassifier(alpha=float(C), class_weight=cw,
                                                 solver=str(params.get("ridge_solver", "lsqr")),
                                                 tol=float(params.get("ridge_tol", 1e-4)),
                                                 max_iter=params.get("ridge_max_iter")))])
    if name == "svm":
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", LinearSVC(C=C, class_weight=cw, max_iter=int(params.get("svm_max_iter", 50000)), tol=1e-4))])
    if name == "lda":
        # Fisher LDA: w = Sigma^-1 (mu_1 - mu_0) with Ledoit-Wolf ("auto") or fixed shrinkage of the pooled covariance.
        shrink = "auto" if C is None or C == "auto" else float(C)
        return Pipeline([("scaler", StandardScaler()), ("clf", ShrunkLDA(shrinkage=shrink))])
    if name == "dim":
        return DiffMeans(standardize=bool(params.get("dim_standardize", False)))
    raise ValueError(name)


def fit_score(name: str, C: float | None, params: dict[str, Any], X_tr, y_tr, X_ev_list: list[np.ndarray]) -> tuple[Any, list[np.ndarray]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf = make_probe(name, C, params).fit(X_tr, y_tr)
    return clf, [clf.decision_function(X) for X in X_ev_list]


@dataclass
class Selection:
    probe: str
    layer: int
    C: float | None
    val_auroc: float
    val_grid: dict[int, dict[str, float]]      # layer -> {str(C): val auroc}
    best_C_per_layer: dict[int, float | None] = field(default_factory=dict)
    val_curve: dict[int, float] = field(default_factory=dict)   # layer -> best val auroc


def select_layer_and_C(name: str, params: dict[str, Any], X_tr_fn: Callable[[int], np.ndarray], y_tr: np.ndarray,
                       X_va_fn: Callable[[int], np.ndarray], y_va: np.ndarray, layers: list[int],
                       save_dir: Path | None = None, extra_meta: dict[str, Any] | None = None) -> Selection:
    if name == "dim":
        C_grid = [None]
    elif name == "lda":
        C_grid = list(params.get("lda_shrinkage_grid", ["auto", 0.1, 0.3, 0.5, 0.7, 0.9]))
    elif name == "ridge":
        C_grid = [float(alpha) for alpha in params.get(
            "ridge_alpha_grid", [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0,
                                  10000.0, 100000.0, 1000000.0])]
    else:
        C_grid = [float(c) for c in params["C_grid"]]
    grid: dict[int, dict[str, float]] = {}
    best_C: dict[int, float | None] = {}
    curve: dict[int, float] = {}
    for l in layers:
        X_tr, X_va = X_tr_fn(l), X_va_fn(l)
        grid[l] = {}
        fits = {}
        for C in C_grid:
            clf, (s_va,) = fit_score(name, C, params, X_tr, y_tr, [X_va])
            grid[l][str(C)] = float(roc_auc_score(y_va, s_va))
            fits[str(C)] = clf
        bC = max(C_grid, key=lambda c: grid[l][str(c)])
        best_C[l] = bC
        curve[l] = grid[l][str(bC)]
        log.info("%s layer %d: best C=%s val AUROC=%.4f", name, l, bC, curve[l])
        if save_dir is not None:
            save_probe(fits[str(bC)], save_dir / name / f"layer_{l:02d}.npz",
                       {"probe": name, "layer": l, "C": bC, "val_auroc": curve[l], "val_grid": grid[l],
                        "n_train": int(len(y_tr)), "n_val": int(len(y_va)), **(extra_meta or {})})
    best_layer = max(layers, key=lambda l: (curve[l], -l))
    return Selection(probe=name, layer=best_layer, C=best_C[best_layer], val_auroc=curve[best_layer],
                     val_grid=grid, best_C_per_layer=best_C, val_curve=curve)

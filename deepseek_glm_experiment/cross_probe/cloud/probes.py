"""Four-arm pooled probe experiment for cloud_v3 models."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..pooled_transfer import (DATASETS, PooledData, dataset_macro_auc, equal_dataset_weights,
                               macro_ba_threshold, organic_question_sample)
from ..transfer_experiments import LinearProbe, fit_linear, fit_linear_candidates, grid_params
from ..transfer_metrics import full_metrics
from ..vendor.io_utils import read_json, write_json

log = logging.getLogger(__name__)

ARMS = {
    "plain_inorganic": {"variant": "basic", "organic": False, "source_arm": None},
    "same_instruction": {"variant": "crossed", "organic": False, "source_arm": None},
    "plain_inorganic_plus_organic": {
        "variant": "basic", "organic": True, "source_arm": "plain_inorganic"
    },
    "same_instruction_plus_organic": {
        "variant": "crossed", "organic": True, "source_arm": "same_instruction"
    },
}


def binary_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Tie-aware Mann-Whitney AUROC without estimator-validation overhead."""
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    pos = int(y.sum())
    neg = len(y) - pos
    if not pos or not neg:
        raise ValueError("AUROC needs both classes")
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.arange(1, len(y) + 1, dtype=np.float64)
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_score)) + 1]
    ends = np.r_[starts[1:], len(y)]
    for start, end in zip(starts, ends):
        if end - start > 1:
            ranks[start:end] = (start + 1 + end) / 2.0
    rank_sum = ranks[y[order] == 1].sum()
    return float((rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def arm_training_data(data: PooledData, register: str, arm: str, seed: int | None,
                      organic_fraction: float, organic_mass: float, layer: int):
    spec = ARMS[arm]
    source = data.source_for(register, spec["variant"])
    train = source.loc[source.split == "train"]
    Xs, ys = data.X(train, layer, "source"), train.label.to_numpy(dtype=np.int8)
    ws = equal_dataset_weights(train, "source", spec["variant"])
    if not spec["organic"]:
        return Xs, ys, ws * len(train), None
    if seed is None:
        raise ValueError("mixed arms require a sampling seed")
    chosen = organic_question_sample(data.organic, organic_fraction, seed)
    Xo, yo = data.X(chosen, layer, "organic"), chosen.label.to_numpy(dtype=np.int8)
    wo = equal_dataset_weights(chosen, "organic")
    X = np.concatenate((Xs, Xo))
    y = np.concatenate((ys, yo))
    weights = np.concatenate(((1.0 - organic_mass) * ws, organic_mass * wo)) * len(X)
    return X, y, weights, chosen


def macro_metrics(rows: pd.DataFrame, scores: np.ndarray, threshold: float) -> dict[str, float]:
    values = []
    by_dataset = {}
    for dataset in DATASETS:
        mask = rows.dataset.to_numpy() == dataset
        metric = full_metrics(rows.loc[mask, "label"].to_numpy(), scores[mask], threshold)
        by_dataset[dataset] = metric
        values.append(metric)
    keys = ("auroc", "auprc", "precision", "recall", "fpr", "fnr", "balanced_accuracy",
            "f1", "truth_precision", "truth_recall", "hallucination_precision", "hallucination_recall")
    return {"equal_dataset_macro": {k: float(np.mean([m[k] for m in values])) for k in keys},
            "by_dataset": by_dataset}


def question_bootstrap_ci(rows: pd.DataFrame, scores: np.ndarray, repeats: int, seed: int) -> list[float]:
    """Equal-dataset AUROC CI, resampling question clusters within each dataset."""
    rng = np.random.RandomState(seed)
    samples = []
    frame = rows.reset_index(drop=True)
    scores = np.asarray(scores)
    grouped = {(d, q): np.flatnonzero((frame.dataset.to_numpy() == d) & (frame.question_id.to_numpy() == q))
               for d in DATASETS for q in frame.loc[frame.dataset == d, "question_id"].unique()}
    qids = {d: sorted(frame.loc[frame.dataset == d, "question_id"].unique()) for d in DATASETS}
    for _ in range(repeats):
        per_dataset = []
        for dataset in DATASETS:
            draw = rng.choice(qids[dataset], size=len(qids[dataset]), replace=True)
            idx = np.concatenate([grouped[(dataset, q)] for q in draw])
            y = frame.iloc[idx].label.to_numpy()
            if len(np.unique(y)) != 2:
                break
            per_dataset.append(binary_auc(y, scores[idx]))
        if len(per_dataset) == len(DATASETS):
            samples.append(float(np.mean(per_dataset)))
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))] if samples else [float("nan")] * 2


def _score_bundle(data: PooledData, model: LinearProbe, layer: int, source: pd.DataFrame,
                  threshold: float, ci_repeats: int, ci_seed: int) -> tuple[dict, dict[str, np.ndarray]]:
    source_test = source.loc[source.split == "test"]
    organic_test = data.organic.loc[data.organic.split == "test"]
    source_scores = model.score(data.X(source_test, layer, "source"))
    organic_scores = model.score(data.X(organic_test, layer, "organic"))
    metrics = {
        "source_test": macro_metrics(source_test, source_scores, threshold),
        "organic_test": macro_metrics(organic_test, organic_scores, threshold),
        "organic_test_auroc_ci95": question_bootstrap_ci(organic_test, organic_scores, ci_repeats, ci_seed),
    }
    arrays = {
        "source_score": source_scores.astype(np.float32),
        "source_label": source_test.label.to_numpy(dtype=np.int8),
        "source_dataset": source_test.dataset.to_numpy(dtype=str),
        "source_question_id": source_test.question_id.to_numpy(dtype=str),
        "organic_score": organic_scores.astype(np.float32),
        "organic_label": organic_test.label.to_numpy(dtype=np.int8),
        "organic_dataset": organic_test.dataset.to_numpy(dtype=str),
        "organic_question_id": organic_test.question_id.to_numpy(dtype=str),
    }
    return metrics, arrays


def fit_arm(data: PooledData, root: Path, probe: str, register: str, arm: str,
            seed: int | None, layers: list[int], organic_fraction: float, organic_mass: float,
            progress: Callable[[dict], None] | None = None) -> dict[str, Any]:
    spec = ARMS[arm]
    seed_name = "fixed" if seed is None else f"seed_{seed}"
    out = root / "models" / probe / register / arm / seed_name
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "result.json"
    if result_path.exists() and (out / "scores.npz").exists() and (out / "selected_probe.npz").exists():
        cached = read_json(result_path)
        expected_mode = "inherit_source" if spec["organic"] else "source_validation_search"
        if cached.get("layer_selection_mode") == expected_mode:
            return cached
    source = data.source_for(register, spec["variant"])
    source_val = source.loc[source.split == "validation"]
    organic_val = data.organic.loc[data.organic.split == "validation"]
    params = data.cfg["probe_params"]
    candidates = grid_params(probe, params)
    grid_path = out / "grid.json"
    inherited = None
    if spec["organic"]:
        base_arm = str(spec["source_arm"])
        base_path = root / "models" / probe / register / base_arm / "fixed" / "result.json"
        if not base_path.exists():
            raise RuntimeError(f"mixed arm {arm} requires completed source arm {base_arm}: {base_path}")
        inherited = read_json(base_path)
        inherited_layer, inherited_param = int(inherited["layer"]), inherited["param"]
        fit_layers = [inherited_layer]
        expected_grid = {"layer_policy": "inherit_source", "source_arm": base_arm,
                         "inherited_layer": inherited_layer, "inherited_param": inherited_param}
        prior = read_json(grid_path) if grid_path.exists() else {}
        grid = prior if all(prior.get(k) == v for k, v in expected_grid.items()) else {
            **expected_grid, "layers": {}
        }
    else:
        fit_layers = layers
        grid = read_json(grid_path) if grid_path.exists() else {
            "layer_policy": "source_validation_search", "layers": {}
        }
    chosen = None
    for layer in fit_layers:
        model_path = out / f"layer_{layer:02d}.npz"
        if str(layer) in grid["layers"] and model_path.exists():
            continue
        X, y, weights, chosen = arm_training_data(data, register, arm, seed, organic_fraction,
                                                  organic_mass, layer)
        evaluation_rows = organic_val if spec["organic"] else source_val
        evaluation_X = data.X(evaluation_rows, layer, "organic" if spec["organic"] else "source")
        if spec["organic"]:
            # Match the retained Qwen/GPT-OSS adaptation: inherit both the
            # source-selected layer and source-selected hyperparameter.
            model = fit_linear(probe, inherited_param, X, y, weights, params)
            auc = dataset_macro_auc(evaluation_rows, model.score(evaluation_X))
            best = (auc, inherited_param, model)
            candidate_values = {str(inherited_param): auc}
        else:
            best = None
            candidate_values = {}
            for param, model in fit_linear_candidates(probe, candidates, X, y, weights, params):
                auc = dataset_macro_auc(evaluation_rows, model.score(evaluation_X))
                candidate_values[str(param)] = auc
                if best is None or auc > best[0]:
                    best = (auc, param, model)
            assert best is not None
        best[2].save(model_path)
        grid["layers"][str(layer)] = {"selection_auroc": best[0], "param": best[1],
                                      "candidate_auroc": candidate_values}
        write_json(grid_path, grid)
        if progress:
            progress({"probe": probe, "register": register, "arm": arm, "seed": seed,
                      "layer": layer, "layers_complete": len(grid["layers"]), "layers_total": len(fit_layers),
                      "layer_policy": grid["layer_policy"]})
    selected_layer = (int(inherited["layer"]) if inherited is not None else
                      max(fit_layers, key=lambda layer: (grid["layers"][str(layer)]["selection_auroc"], -layer)))
    selected = grid["layers"][str(selected_layer)]
    model = LinearProbe.load(out / f"layer_{selected_layer:02d}.npz")
    model.save(out / "selected_probe.npz")
    source_val_scores = model.score(data.X(source_val, selected_layer, "source"))
    threshold = macro_ba_threshold(source_val, source_val_scores)
    ci_cfg = data.cfg["confidence_intervals"]
    metrics, arrays = _score_bundle(data, model, selected_layer, source, threshold,
                                    int(ci_cfg["samples"]), int(ci_cfg["seed"]) + (seed or 0))
    chosen = (organic_question_sample(data.organic, organic_fraction, int(seed))
              if spec["organic"] else None)

    def counts(rows: pd.DataFrame) -> dict[str, dict[str, int]]:
        return {dataset: {"truth": int(((rows.dataset == dataset) & (rows.label == 0)).sum()),
                          "hallucination": int(((rows.dataset == dataset) & (rows.label == 1)).sum()),
                          "questions": int(rows.loc[rows.dataset == dataset, "question_id"].nunique())}
                for dataset in DATASETS}

    result = {
        "model": data.model, "probe": probe, "register": register, "arm": arm,
        "source_variant": spec["variant"], "seed": seed, "layer": selected_layer,
        "param": selected["param"], "selection_auroc": selected["selection_auroc"],
        "selection_distribution": (f"inherited_from_{spec['source_arm']}" if spec["organic"]
                                   else "source_validation"),
        "layer_selection_mode": "inherit_source" if spec["organic"] else "source_validation_search",
        "source_arm": spec["source_arm"],
        "organic_validation_auroc": selected["selection_auroc"] if spec["organic"] else None,
        "threshold": threshold, "threshold_distribution": "source_validation_balanced_accuracy",
        "organic_fraction": organic_fraction if spec["organic"] else 0.0,
        "organic_mass": organic_mass if spec["organic"] else 0.0,
        "n_organic_questions": int(chosen.question_id.nunique()) if chosen is not None else 0,
        "n_organic_rows": len(chosen) if chosen is not None else 0,
        "n_source_train_rows": int((source.split == "train").sum()),
        "sample_counts": {"source_train": counts(source.loc[source.split == "train"]),
                          "organic_train": counts(chosen) if chosen is not None else {},
                          "source_test": counts(source.loc[source.split == "test"]),
                          "organic_test": counts(data.organic.loc[data.organic.split == "test"])},
        "metrics": metrics,
    }
    np.savez_compressed(out / "scores.npz", **arrays)
    selection_rule = (f"inherit layer and hyperparameter from {spec['source_arm']}; report organic-validation AUROC"
                      if spec["organic"] else
                      "maximum equal-dataset source-validation AUROC; ties choose shallower layer")
    write_json(out / "selection.json", {**result, "metrics": None,
                                         "selection_rule": selection_rule,
                                         "layers": fit_layers})
    write_json(result_path, result)
    return result


def run_experiment(data: PooledData, root: Path, probes: tuple[str, ...], registers: tuple[str, ...],
                   arms: tuple[str, ...], seeds: tuple[int, ...], organic_fraction: float,
                   organic_mass: float, layers: list[int] | None = None,
                   progress: Callable[[dict], None] | None = None) -> list[dict[str, Any]]:
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise ValueError(f"unknown arms: {sorted(unknown)}")
    n_layers = data.runs[DATASETS[0]].layers
    if any(run.layers != n_layers for run in data.runs.values()):
        raise ValueError("activation layer count differs across datasets")
    layers = list(range(n_layers)) if layers is None else layers
    for arm in arms:
        source_arm = ARMS[arm]["source_arm"]
        if source_arm is not None and source_arm not in arms:
            raise ValueError(f"mixed arm {arm} requires its source arm {source_arm}")
    # Source fits must exist before their mixed adaptations even if a caller
    # supplies the configured arms in a different order.
    ordered_arms = tuple(a for a in arms if not ARMS[a]["organic"]) + tuple(
        a for a in arms if ARMS[a]["organic"]
    )
    results = []
    for probe in probes:
        for register in registers:
            for arm in ordered_arms:
                arm_seeds = seeds if ARMS[arm]["organic"] else (None,)
                for seed in arm_seeds:
                    results.append(fit_arm(data, root, probe, register, arm, seed, layers,
                                           organic_fraction, organic_mass, progress))
    write_json(root / "results.json", results)
    return results

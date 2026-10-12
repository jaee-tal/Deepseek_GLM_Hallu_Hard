"""Pooled GPT-OSS transfer with crossed, same-instruction, and mixed sources.

Fit/search never reads test activations. The report phase alone evaluates held-out data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .pooled_transfer import (BUDGETS, DATASETS, MASSES, PROBES, REGISTERS, SEEDS,
                              PooledData, dataset_macro_auc, equal_dataset_weights,
                              fingerprint, organic_question_sample)
from .transfer_experiments import LinearProbe, fit_linear, fit_linear_candidates, grid_params
from .vendor.io_utils import read_json, write_json

SOURCE_VARIANT = "crossed"
SOURCE_VARIANTS = ("basic", "crossed")


def _source_signature(data: PooledData, register: str, variant: str) -> str:
    rows = data.source_for(register, variant)
    return fingerprint({"model": data.cfg.model(data.model)["checkpoint"], "register": register,
                        "variant": variant, "pooled": data.signature(register, variant),
                        "rows": rows[["conversation_id", "label", "split"]].to_dict("list")})


def _source_fit_signature(data: PooledData, probe: str, register: str, variant: str) -> str:
    """Include probe-specific hyperparameter candidates in the cache signature."""
    base = _source_signature(data, register, variant)
    if probe != "ridge":
        return base
    return fingerprint({"data": base, "probe": probe,
                        "candidates": grid_params(probe, data.cfg["probe_params"])})


def _source_path(root: Path, probe: str, register: str, variant: str) -> Path:
    return root / "models" / probe / register / variant / "source"


def _adapted_path(root: Path, probe: str, register: str, variant: str) -> Path:
    return root / "models" / probe / register / variant / "adapted"


def _validate_source(rows: pd.DataFrame, register: str, variant: str) -> None:
    if variant not in SOURCE_VARIANTS:
        raise ValueError(variant)
    cells = ["condition", "label"] if variant == "crossed" else ["label"]
    required = 4 if variant == "crossed" else 2
    for dataset in DATASETS:
        for split in ("train", "validation", "test"):
            part = rows.loc[(rows.dataset == dataset) & (rows.split == split)]
            if part.groupby(cells, dropna=False).ngroups != required:
                raise ValueError(f"{dataset}/{split}/{register}/{variant}: expected {required} nonempty source cells")


def fit_source(data: PooledData, root: Path, probe: str, register: str, variant: str,
               layers: list[int] | None = None) -> dict[str, Any]:
    rows = data.source_for(register, variant)
    _validate_source(rows, register, variant)
    train, val = (rows.loc[rows.split == split] for split in ("train", "validation"))
    out = _source_path(root, probe, register, variant)
    out.mkdir(parents=True, exist_ok=True)
    signature = _source_fit_signature(data, probe, register, variant)
    grid_path = out / "grid.json"
    prior = read_json(grid_path) if grid_path.exists() else {}
    grid = prior if prior.get("signature") == signature else {"signature": signature, "layers": {}}
    all_layers = data.runs[DATASETS[0]].layers
    if any(run.layers != all_layers for run in data.runs.values()):
        raise ValueError("GPT-OSS activation layer counts differ across datasets")
    chosen_layers = range(all_layers) if layers is None else layers
    params = data.cfg["probe_params"]
    candidates = grid_params(probe, params)
    y = train.label.to_numpy()
    w = equal_dataset_weights(train, "source", variant) * len(train)
    for layer in chosen_layers:
        path = out / f"layer_{layer:02d}.npz"
        if str(layer) in grid["layers"] and path.exists():
            continue
        X, Xv = data.X(train, layer, "source"), data.X(val, layer, "source")
        best = None
        for param, model in fit_linear_candidates(probe, candidates, X, y, w, params):
            auc = dataset_macro_auc(val, model.score(Xv))
            if best is None or auc > best[0]:
                best = (auc, param, model)
        assert best is not None
        best[2].save(path)
        grid["layers"][str(layer)] = {"validation_macro_auroc": best[0], "param": best[1]}
        write_json(grid_path, grid)
        print(f"source {probe}/{register}/{variant} layer {layer:02d}: validation AUROC {best[0]:.4f}", flush=True)
    if len(grid["layers"]) != all_layers:
        return {"complete": False, "layers_done": len(grid["layers"]), "n_layers": all_layers}
    layer = max(range(all_layers), key=lambda i: (grid["layers"][str(i)]["validation_macro_auroc"], -i))
    selected = {"complete": True, "signature": signature, "layer": layer,
                "param": grid["layers"][str(layer)]["param"],
                "validation_macro_auroc": grid["layers"][str(layer)]["validation_macro_auroc"]}
    write_json(out / "selection.json", selected)
    return selected


def fit_adaptation(data: PooledData, root: Path, probe: str, register: str, variant: str,
                   selection: dict[str, Any], budgets=BUDGETS, masses=MASSES,
                   seeds=SEEDS) -> list[dict[str, Any]]:
    layer, param = int(selection["layer"]), selection["param"]
    source = data.source_for(register, variant)
    train = source.loc[source.split == "train"]
    val = data.organic.loc[data.organic.split == "validation"]
    Xsrc, ysrc = data.X(train, layer, "source"), train.label.to_numpy()
    wsrc = equal_dataset_weights(train, "source", variant)
    Xval = data.X(val, layer, "organic")
    out = _adapted_path(root, probe, register, variant)
    out.mkdir(parents=True, exist_ok=True)
    trials = []
    for seed in seeds:
        for budget in budgets:
            chosen = organic_question_sample(data.organic, budget, seed)
            Xorg, yorg = data.X(chosen, layer, "organic"), chosen.label.to_numpy()
            worg = equal_dataset_weights(chosen, "organic")
            for mass in masses:
                stem = f"seed{seed}_budget{budget:g}_organic{mass:g}"
                path, meta = out / f"{stem}.npz", out / f"{stem}.json"
                signature = fingerprint({"source": selection, "variant": variant,
                                         "organic_ids": chosen.conversation_id.tolist(),
                                         "organic_labels": yorg.tolist(), "seed": seed,
                                         "budget": budget, "mass": mass})
                old = read_json(meta) if meta.exists() else None
                if old and old.get("signature") == signature and path.exists():
                    trials.append(old)
                    continue
                X = np.concatenate((Xsrc, Xorg))
                y = np.concatenate((ysrc, yorg))
                weights = np.concatenate(((1 - mass) * wsrc, mass * worg)) * len(train)
                model = fit_linear(probe, param, X, y, weights, data.cfg["probe_params"])
                record = {"probe": probe, "register": register, "variant": variant, "seed": seed,
                          "budget": budget, "mass": mass, "signature": signature,
                          "layer": layer, "param": param,
                          "n_organic_questions": int(chosen.question_id.nunique()),
                          "n_organic_answers": len(chosen),
                          "organic_validation_macro_auroc": dataset_macro_auc(val, model.score(Xval))}
                model.save(path)
                write_json(meta, record)
                trials.append(record)
                print(f"adapted {probe}/{register}/{variant} seed {seed} budget {budget:g} mass {mass:g}: "
                      f"validation AUROC {record['organic_validation_macro_auroc']:.4f}", flush=True)
    return trials


def choose_mix(trials: list[dict[str, Any]], seeds=SEEDS) -> tuple[float, float, pd.DataFrame]:
    frame = pd.DataFrame(trials)
    if frame.empty:
        raise ValueError("empty adaptation search")
    grouped = frame.groupby(["budget", "mass"]).agg(
        organic_validation_macro_auroc=("organic_validation_macro_auroc", "mean"),
        n_fits=("organic_validation_macro_auroc", "size")).reset_index()
    if not (grouped.n_fits == len(REGISTERS) * len(seeds)).all():
        raise ValueError("incomplete mix search across registers and seeds")
    selected = grouped.sort_values(["organic_validation_macro_auroc", "budget", "mass"],
                                   ascending=[False, True, True]).iloc[0]
    return float(selected.budget), float(selected.mass), grouped


def fit_all(data: PooledData, root: Path, probes: tuple[str, ...], layers: list[int] | None = None) -> None:
    """Fit both inorganic sources and each source's organic-adaptation grid."""
    for probe in probes:
        trials: dict[str, list[dict[str, Any]]] = {variant: [] for variant in SOURCE_VARIANTS}
        complete = {variant: True for variant in SOURCE_VARIANTS}
        for register in REGISTERS:
            selections = {
                variant: fit_source(data, root, probe, register, variant, layers)
                for variant in SOURCE_VARIANTS
            }
            for variant, selection in selections.items():
                if not selection["complete"]:
                    complete[variant] = False
                    print(f"source grid incomplete: {probe}/{register}/{variant}", flush=True)
                else:
                    trials[variant].extend(
                        fit_adaptation(data, root, probe, register, variant, selection))
        search = root / "search" / probe
        search.mkdir(parents=True, exist_ok=True)
        for variant in SOURCE_VARIANTS:
            if not complete[variant]:
                continue
            budget, mass, grid = choose_mix(trials[variant])
            suffix = "" if variant == SOURCE_VARIANT else f"_{variant}"
            grid.to_csv(search / f"mix_grid{suffix}.csv", index=False)
            write_json(search / f"selected_mix{suffix}.json",
                       {"probe": probe, "source_variant": variant, "budget": budget, "mass": mass,
                        "selection_rule": "equal-dataset organic-validation AUROC across registers and seeds"})
            print(f"selected {probe}/{variant}: {budget:g} organic questions, "
                  f"{mass:g} organic weight", flush=True)

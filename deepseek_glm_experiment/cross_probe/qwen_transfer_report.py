"""Canonical Qwen report for crossed, same-instruction, and mixed transfer."""
from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import Config
from .pooled_transfer import BUDGETS, DATASETS, MASSES, REGISTERS, SEEDS
from .transfer_experiments import (LinearProbe, balanced_accuracy_threshold, load_run_data,
                                   adaptation_dir, sampled_organic_questions, source_view)
from .transfer_metrics import METRICS, PROBE_LABELS, full_metrics, macro_table, pooled_confusions
from .vendor.io_utils import write_json

PROBES = ("dim", "logistic", "lda")
METHODS = ("inorganic_only", "same_instruction", "inorganic_plus_organic",
           "same_instruction_plus_organic")
LABELS = {"inorganic_only": "Inorganic only", "same_instruction": "Same instruction",
          "inorganic_plus_organic": "Inorganic + organic",
          "same_instruction_plus_organic": "Same instruction + organic"}
SOURCE_METHODS = {"basic": "inorganic_only", "crossed": "same_instruction"}
MIXED_METHODS = {"basic": "inorganic_plus_organic",
                 "crossed": "same_instruction_plus_organic"}
COLORS = {"source_truth": "#2374B5", "source_hallucination": "#CC4141",
          "organic_truth": "#248C57", "organic_hallucination": "#E6922E"}
HIST_LABELS = {"source_truth": "Inorganic truth", "source_hallucination": "Inorganic hallucination",
               "organic_truth": "Organic truth", "organic_hallucination": "Organic hallucination"}


def _method(variant: str, budget: float, mass: float) -> str:
    return f"mixed_{variant}_{budget:g}_{mass:g}"


def collect(cfg: Config, tag: str, probe: str):
    runs = {dataset: load_run_data(cfg, dataset, tag=tag, model="qwen3_8b") for dataset in DATASETS}
    organic_records, source_records, thresholds, validation = [], [], [], []
    score_packs = {}
    layer_rows = []
    for dataset, data in runs.items():
        org_val = data.organic_rows.loc[data.organic_rows.split == "validation"]
        org_test = data.organic_rows.loc[data.organic_rows.split == "test"]
        yov, yo = org_val.label.to_numpy(), org_test.label.to_numpy()
        for register in REGISTERS:
            root = data.run_dir / "results" / "transfer_experiments"
            source_setups = {}
            for variant, method in SOURCE_METHODS.items():
                src = source_view(data.source_rows, register, variant)
                val, test = (src.loc[src.split == split] for split in ("validation", "test"))
                fit_dir = root / "outcome" / register / variant / probe
                selection = json.loads((fit_dir / "selection.json").read_text())
                layer = int(selection["layer"])
                grid = json.loads((fit_dir / "grid.json").read_text())
                for key, entry in grid["layers"].items():
                    layer_rows.append({"dataset": dataset, "register": register, "probe": probe,
                                       "method": method, "source_variant": variant, "layer": int(key),
                                       "validation_macro_auroc": entry["best_val_macro_auc"],
                                       "selected": int(key) == layer})
                source_setups[method] = (variant, val, test, fit_dir, layer)

            # Each inorganic method selects its own layer and threshold from its
            # own source validation set.
            for method in ("inorganic_only", "same_instruction"):
                variant, val, test, fit_dir, layer = source_setups[method]
                model = LinearProbe.load(fit_dir / f"layer_{layer:02d}_best.npz")
                Xv, Xs = data.source_X(val, layer), data.source_X(test, layer)
                Xov, Xo = data.organic_X(org_val, layer), data.organic_X(org_test, layer)
                sv, ss, sov, so = model.score(Xv), model.score(Xs), model.score(Xov), model.score(Xo)
                yl, yt = val.label.to_numpy(), test.label.to_numpy()
                threshold = balanced_accuracy_threshold(yl, sv)
                stem = {"probe": probe, "dataset": dataset, "register": register,
                        "method": method, "seed": None, "threshold": threshold, "layer": layer,
                        "source_variant": variant, "budget": 0.0, "organic_mass": 0.0}
                organic_records.append({**stem, **full_metrics(yo, so, threshold)})
                source_records.append({**stem, **full_metrics(yt, ss, threshold)})
                thresholds.append({**stem, "source_validation_n": len(yl),
                                   "source_validation_balanced_accuracy":
                                       full_metrics(yl, sv, threshold)["balanced_accuracy"]})
                validation.append({"probe": probe, "dataset": dataset, "register": register,
                                   "method": method, "seed": None, "budget": 0.0,
                                   "organic_mass": 0.0,
                                   "organic_validation_auroc": float(roc_auc_score(yov, sov))})
                score_packs[(dataset, register, method, None)] = {
                    "source_y": yt, "source_score": ss, "organic_y": yo, "organic_score": so,
                    "organic_qids": org_test.question_id.to_numpy(), "threshold": threshold,
                    "scale": float(np.std(sv)),
                }

            for variant, base_method in SOURCE_METHODS.items():
                _, val, test, _, layer = source_setups[base_method]
                Xv, Xs = data.source_X(val, layer), data.source_X(test, layer)
                Xov, Xo = data.organic_X(org_val, layer), data.organic_X(org_test, layer)
                adapt = adaptation_dir(data.run_dir, register, probe, variant)
                yl, yt = val.label.to_numpy(), test.label.to_numpy()
                for budget in BUDGETS:
                    for mass in MASSES:
                        for seed in SEEDS:
                            method = _method(variant, budget, mass)
                            stem_name = f"seed{seed}_budget{budget:g}_organic{mass:g}"
                            model = LinearProbe.load(adapt / f"{stem_name}.npz")
                            sv, ss, sov, so = (model.score(Xv), model.score(Xs),
                                               model.score(Xov), model.score(Xo))
                            threshold = balanced_accuracy_threshold(yl, sv)
                            stem = {"probe": probe, "dataset": dataset, "register": register,
                                    "method": method, "seed": seed, "threshold": threshold,
                                    "layer": layer, "source_variant": variant,
                                    "budget": budget, "organic_mass": mass}
                            organic_records.append({**stem, **full_metrics(yo, so, threshold)})
                            source_records.append({**stem, **full_metrics(yt, ss, threshold)})
                            thresholds.append({**stem, "source_validation_n": len(yl),
                                               "source_validation_balanced_accuracy":
                                                   full_metrics(yl, sv, threshold)["balanced_accuracy"]})
                            validation.append({"probe": probe, "dataset": dataset,
                                               "register": register, "method": method,
                                               "source_variant": variant, "seed": seed,
                                               "budget": budget, "organic_mass": mass,
                                               "organic_validation_auroc":
                                                   float(roc_auc_score(yov, sov))})
                            score_packs[(dataset, register, method, seed)] = {
                                "source_y": yt, "source_score": ss, "organic_y": yo,
                                "organic_score": so, "organic_qids": org_test.question_id.to_numpy(),
                                "threshold": threshold, "scale": float(np.std(sv)),
                            }
    return (pd.DataFrame(organic_records), pd.DataFrame(source_records), pd.DataFrame(thresholds),
            pd.DataFrame(validation), score_packs, pd.DataFrame(layer_rows), runs)


def select_mix(validation: pd.DataFrame, variant: str) -> tuple[float, float, pd.DataFrame]:
    mixed = validation.loc[validation.source_variant.eq(variant)
                           & validation.method.str.startswith(f"mixed_{variant}_")]
    grid = mixed.groupby(["budget", "organic_mass"], as_index=False).agg(
        organic_validation_auroc=("organic_validation_auroc", "mean"), n_fits=("organic_validation_auroc", "size"))
    expected = len(DATASETS) * len(REGISTERS) * len(SEEDS)
    if not (grid.n_fits == expected).all(): raise ValueError("incomplete Qwen mix grid")
    best = grid.sort_values(["organic_validation_auroc", "budget", "organic_mass"],
                            ascending=[False, True, True]).iloc[0]
    return float(best.budget), float(best.organic_mass), grid


def main_records(records: pd.DataFrame, selections: dict[str, tuple[float, float]]) -> pd.DataFrame:
    mapping = {_method(variant, *choice): MIXED_METHODS[variant]
               for variant, choice in selections.items()}
    keep = set(SOURCE_METHODS.values()) | set(mapping)
    out = records.loc[records.method.isin(keep)].copy()
    out["method"] = out["method"].replace(mapping)
    return out


def histogram_data(packs: dict, selections: dict[str, tuple[float, float]]) -> dict:
    out = defaultdict(list)
    for register in REGISTERS:
        keys = [(method, method, None) for method in SOURCE_METHODS.values()]
        for variant, choice in selections.items():
            keys.extend((MIXED_METHODS[variant], _method(variant, *choice), seed) for seed in SEEDS)
        for label, raw, seed in keys:
            for dataset in DATASETS:
                p = packs[(dataset, register, raw, seed)]
                if p["scale"] <= 0: raise ValueError("constant source-validation scores")
                ss = (p["source_score"] - p["threshold"]) / p["scale"]
                so = (p["organic_score"] - p["threshold"]) / p["scale"]
                out[(label, register)].append({
                    "source_truth": ss[p["source_y"] == 0],
                    "source_hallucination": ss[p["source_y"] == 1],
                    "organic_truth": so[p["organic_y"] == 0],
                    "organic_hallucination": so[p["organic_y"] == 1],
                })
    return out


def paired_bootstrap(packs: dict, selections: dict[str, tuple[float, float]],
                     repeats: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed); rows = []

    def metrics(p, idx):
        y, score = p["organic_y"][idx], p["organic_score"][idx]; pred = score >= p["threshold"]
        recall = float(np.mean(pred[y == 1])); fpr = float(np.mean(pred[y == 0]))
        return {"auroc": float(roc_auc_score(y, score)), "recall": recall, "fpr": fpr,
                "balanced_accuracy": (recall + 1 - fpr) / 2}

    for register in REGISTERS:
        alternatives_to_compare = [
            ("same_instruction", [("same_instruction", None)]),
            ("inorganic_plus_organic",
             [(_method("basic", *selections["basic"]), s) for s in SEEDS]),
            ("same_instruction_plus_organic",
             [(_method("crossed", *selections["crossed"]), s) for s in SEEDS]),
        ]
        for alternative, raw_keys in alternatives_to_compare:
            blocks = []
            for dataset in DATASETS:
                base = packs[(dataset, register, "inorganic_only", None)]
                alternatives = [packs[(dataset, register, method, fit_seed)]
                                for method, fit_seed in raw_keys]
                unique, inverse = np.unique(base["organic_qids"], return_inverse=True)
                blocks.append((base, alternatives,
                               [np.flatnonzero(inverse == i) for i in range(len(unique))]))
            point = {k: [] for k in ("auroc", "recall", "fpr", "balanced_accuracy")}
            for base, alternatives, _ in blocks:
                idx = np.arange(len(base["organic_y"])); b = metrics(base, idx)
                a = {k: np.mean([metrics(p, idx)[k] for p in alternatives]) for k in point}
                for k in point: point[k].append(a[k] - b[k])
            draws = {k: [] for k in point}
            while len(draws["auroc"]) < repeats:
                diff = {k: [] for k in point}
                for base, alternatives, groups in blocks:
                    idx = np.concatenate([groups[i] for i in rng.randint(len(groups), size=len(groups))])
                    if len(np.unique(base["organic_y"][idx])) != 2: break
                    b = metrics(base, idx)
                    a = {k: np.mean([metrics(p, idx)[k] for p in alternatives]) for k in point}
                    for k in point: diff[k].append(a[k] - b[k])
                else:
                    for k in draws: draws[k].append(float(np.mean(diff[k])))
            for metric in point:
                low, high = np.percentile(draws[metric], [2.5, 97.5])
                rows.append({"register": register,
                             "comparison": f"{alternative} - inorganic_only",
                             "metric": metric, "difference": float(np.mean(point[metric])),
                             "ci95_low": low, "ci95_high": high, "n_bootstrap": repeats})
    return pd.DataFrame(rows)


def _plt():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cross_probe_matplotlib"))
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "savefig.facecolor": "white"})
    return plt


def plot_matrix(macro: pd.DataFrame, probe: str, out: Path) -> None:
    plt = _plt(); labels = ("AUROC", "AUPRC", "Precision", "Recall", "FPR", "FNR", "Balanced acc.")
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2), sharey=True); cmap = plt.get_cmap("RdYlGn")
    for ax, register in zip(axes, REGISTERS):
        values = macro.loc[macro.register == register].set_index("method").loc[list(METHODS), list(METRICS)].to_numpy()
        colors = values.copy(); colors[:, 4:6] = 1 - colors[:, 4:6]; ax.imshow(colors, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(METRICS)), labels, rotation=38, ha="left"); ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, length=0)
        ax.set_yticks(range(len(METHODS)), [LABELS[m] for m in METHODS]); ax.set_xticks(np.arange(-.5, len(METRICS), 1), minor=True); ax.set_yticks(np.arange(-.5, len(METHODS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=3); ax.tick_params(which="minor", bottom=False, left=False)
        for i in range(len(METHODS)):
            for j in range(len(METRICS)): ax.text(j, i, f"{values[i,j]:.3f}", ha="center", va="center", fontsize=8.7, weight="bold")
        ax.set_title(register.title(), weight="bold"); [s.set_visible(False) for s in ax.spines.values()]
    fig.suptitle(f"{PROBE_LABELS[probe]}: Qwen3-8B transfer to organic answers", fontsize=16)
    fig.text(.5, .035, "Equal-dataset mean; mixed fits average five seeds. Thresholds maximize balanced accuracy on inorganic validation data.", ha="center")
    fig.subplots_adjust(left=.15, right=.99, top=.78, bottom=.19, wspace=.14); fig.savefig(out, dpi=180); plt.close(fig)


def plot_confusions(counts: pd.DataFrame, probe: str, out: Path) -> None:
    plt = _plt(); fig, axes = plt.subplots(len(METHODS), 3, figsize=(12.2, 9.8))
    for row, method in enumerate(METHODS):
        for col, register in enumerate(REGISTERS):
            ax = axes[row,col]; r = counts.loc[(counts.method == method) & (counts.register == register)].iloc[0]
            cm = np.array([[r.tn, r.fp], [r.fn, r.tp]]); ax.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2): ax.text(j, i, f"{cm[i,j]:.0f}", ha="center", va="center", weight="bold")
            if row == 0: ax.set_title(register.title(), weight="bold")
            if col == 0: ax.set_ylabel(f"{LABELS[method]}\nTrue label")
            if row == len(METHODS) - 1: ax.set_xlabel("Predicted label")
            ax.set_xticks([0,1], ["Truth","Hallucination"]); ax.set_yticks([0,1], ["Truth","Hallucination"])
    fig.suptitle(f"{PROBE_LABELS[probe]}: organic-test confusion matrices", fontsize=15)
    fig.tight_layout(rect=(0,0,1,.96)); fig.savefig(out, dpi=180); plt.close(fig)


def plot_histograms(hist: dict, probe: str, out: Path) -> None:
    plt = _plt(); fig, axes = plt.subplots(len(METHODS), 3, figsize=(15, 12.5), sharey="row")
    for row, method in enumerate(METHODS):
        for col, register in enumerate(REGISTERS):
            ax = axes[row,col]; entries = hist[(method,register)]
            arrays = {k: np.concatenate([e[k] for e in entries]) for k in COLORS}; all_scores = np.concatenate(list(arrays.values()))
            lo, hi = np.quantile(all_scores, [.005,.995]); bins = np.linspace(lo, hi, 45); repeat_weight = 1 / len(entries)
            for key in COLORS:
                ax.hist(arrays[key], bins=bins, histtype="step", linewidth=1.8, color=COLORS[key], weights=np.full(len(arrays[key]), repeat_weight), label=HIST_LABELS[key])
            ax.axvline(0, color="#25343D", linestyle="--", linewidth=1)
            if row == 0: ax.set_title(register.title(), weight="bold")
            if col == 0: ax.set_ylabel(f"{LABELS[method]}\nSamples")
            if row == len(METHODS) - 1: ax.set_xlabel("Probe score (threshold-centered, validation-SD units)")
    axes[0,0].legend(fontsize=8); fig.suptitle(f"{PROBE_LABELS[probe]}: score distributions", fontsize=15)
    fig.tight_layout(rect=(0,0,1,.96)); fig.savefig(out, dpi=180); plt.close(fig)


def plot_layers(rows: pd.DataFrame, probe: str, out: Path) -> None:
    plt = _plt(); fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), sharey=True)
    colors = {"pop_qa":"#376A9A", "trivia_qa":"#C66A2B", "hallu_hard":"#4A8C62"}
    styles = {"inorganic_only": "-", "same_instruction": "--"}
    for ax, register in zip(axes, REGISTERS):
        for dataset in DATASETS:
            for method in ("inorganic_only", "same_instruction"):
                part = rows.loc[(rows.register == register) & (rows.dataset == dataset)
                                & (rows.method == method)].sort_values("layer")
                ax.plot(part.layer, part.validation_macro_auroc,
                        label=f"{dataset} / {LABELS[method]}", color=colors[dataset],
                        linestyle=styles[method], linewidth=1.8)
                chosen = part.loc[part.selected].iloc[0]
                ax.scatter([chosen.layer], [chosen.validation_macro_auroc],
                           color=colors[dataset], marker="o" if method == "inorganic_only" else "s", s=35)
        ax.set_title(register.title(), weight="bold"); ax.set_xlabel("Layer (0 = embeddings)"); ax.grid(alpha=.25)
    axes[0].set_ylabel("Inorganic-validation AUROC"); axes[0].legend(fontsize=8)
    fig.suptitle(f"{PROBE_LABELS[probe]}: independent source-layer selection", fontsize=15)
    fig.tight_layout(rect=(0,0,1,.94)); fig.savefig(out, dpi=180); plt.close(fig)


def report(cfg: Config, out: Path, tag: str = "pilot", probes: tuple[str, ...] = PROBES) -> None:
    out.mkdir(parents=True, exist_ok=True); overview = []
    write_json(out / "manifest.json", {"model": "Qwen/Qwen3-8B", "datasets": DATASETS, "registers": REGISTERS,
               "methods": METHODS, "source_variants": ("basic", "crossed"), "seeds": SEEDS,
               "threshold": "per-dataset inorganic-validation balanced accuracy"})
    for probe in probes:
        organic, source, thresholds, validation, packs, layers, runs = collect(cfg, tag, probe)
        if probe == probes[0]:
            sample_counts = []
            for dataset, data in runs.items():
                for split in ("train", "validation", "test"):
                    org = data.organic_rows.loc[data.organic_rows.split == split]
                    sample_counts.append({"dataset": dataset, "split": split, "domain": "organic",
                                          "register": "all", "questions": org.question_id.nunique(),
                                          "answers": len(org), "truth": int((org.label == 0).sum()),
                                          "hallucination": int((org.label == 1).sum())})
                    for register in REGISTERS:
                        for variant in ("basic", "crossed"):
                            src = source_view(data.source_rows, register, variant)
                            part = src.loc[src.split == split]
                            sample_counts.append({"dataset": dataset, "split": split, "domain": "inorganic",
                                                  "source_variant": variant, "register": register,
                                                  "questions": part.question_id.nunique(),
                                                  "answers": len(part), "truth": int((part.label == 0).sum()),
                                                  "hallucination": int((part.label == 1).sum())})
            pd.DataFrame(sample_counts).to_csv(out / "sample_counts.csv", index=False)
        selections, grids = {}, []
        for variant in ("basic", "crossed"):
            budget, mass, grid = select_mix(validation, variant)
            selections[variant] = (budget, mass)
            grid.insert(0, "source_variant", variant); grids.append(grid)
        mix_grid = pd.concat(grids, ignore_index=True)
        org_main, src_main, thr_main = (main_records(frame, selections)
                                        for frame in (organic, source, thresholds))
        org_macro, src_macro = macro_table(org_main), macro_table(src_main); cms = pooled_confusions(org_main)
        target = out / probe; target.mkdir(parents=True, exist_ok=True)
        organic.to_csv(target / "all_organic_transfer_results.csv", index=False)
        macro_table(organic).to_csv(target / "all_organic_transfer_macro.csv", index=False)
        validation.to_csv(target / "all_mix_validation_results.csv", index=False); mix_grid.to_csv(target / "mix_validation_grid.csv", index=False)
        org_main.to_csv(target / "organic_test_per_dataset.csv", index=False); src_main.to_csv(target / "inorganic_test_per_dataset.csv", index=False)
        org_macro.to_csv(target / "organic_test_macro.csv", index=False); src_macro.to_csv(target / "inorganic_test_macro.csv", index=False)
        thr_main.to_csv(target / "validation_thresholds.csv", index=False); cms.to_csv(target / "organic_test_confusion_counts.csv", index=False)
        paired_bootstrap(packs, selections).to_csv(target / "paired_transfer_deltas_ci95.csv", index=False)
        overall = org_macro.groupby(["probe","method"], as_index=False)[list(METRICS)].mean(); overall.to_csv(target / "organic_test_overall.csv", index=False); overview.append(overall)
        training = []
        for variant, (budget, mass) in selections.items():
            for dataset, data in runs.items():
                for seed in SEEDS:
                    chosen = sampled_organic_questions(data.organic_rows, budget, seed)
                    training.append({"probe": probe, "source_variant": variant,
                                     "organic_mass": mass, "seed": seed, "dataset": dataset,
                                     "organic_questions": chosen.question_id.nunique(),
                                     "organic_answers": len(chosen),
                                     "organic_truth": int((chosen.label == 0).sum()),
                                     "organic_hallucination": int((chosen.label == 1).sum())})
        pd.DataFrame(training).to_csv(target / "selected_training_counts.csv", index=False)
        write_json(target / "selected_mix.json", {
            variant: {"budget": budget, "organic_mass": mass}
            for variant, (budget, mass) in selections.items()})
        layers.to_csv(target / "layer_selection_auroc.csv", index=False)
        plot_matrix(org_macro, probe, target / "organic_test_comparison.png"); plot_confusions(cms, probe, target / "organic_test_confusions.png")
        plot_histograms(histogram_data(packs, selections), probe, target / "probe_score_histograms.png")
        plot_layers(layers, probe, target / "layer_selection_auroc.png")
        print(f"reported {probe} -> {target}", flush=True)
    completed = [pd.read_csv(path) for path in sorted(out.glob("*/organic_test_overall.csv"))]
    pd.concat(completed, ignore_index=True).to_csv(out / "transfer_overview.csv", index=False)

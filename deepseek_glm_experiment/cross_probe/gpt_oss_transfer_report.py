"""Held-out transfer report for the pooled GPT-OSS experiment."""
from __future__ import annotations

import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .gpt_oss_transfer import SOURCE_VARIANT, SOURCE_VARIANTS, _adapted_path, _source_path
from .pooled_transfer import (DATASETS, REGISTERS, SEEDS, PooledData,
                              macro_ba_threshold, organic_question_sample)
from .transfer_experiments import LinearProbe
from .transfer_metrics import METRICS, PROBE_LABELS, full_metrics, macro_table, pooled_confusions
from .vendor.io_utils import read_json, write_json

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


def _selected(root: Path, probe: str, variant: str) -> dict:
    suffix = "" if variant == SOURCE_VARIANT else f"_{variant}"
    return read_json(root / "search" / probe / f"selected_mix{suffix}.json")


def collect(data: PooledData, root: Path, probe: str):
    selected = {variant: _selected(root, probe, variant) for variant in SOURCE_VARIANTS}
    org = data.organic.loc[data.organic.split == "test"]
    organic_records, source_records, thresholds = [], [], []
    hist: dict[tuple[str, str], list[dict[str, np.ndarray]]] = defaultdict(list)
    paired, selections = {}, {}
    for register in REGISTERS:
        setups = {}
        for variant, method in SOURCE_METHODS.items():
            src = data.source_for(register, variant)
            val, test = (src.loc[src.split == split] for split in ("validation", "test"))
            base = _source_path(root, probe, register, variant)
            selection = read_json(base / "selection.json")
            selections.setdefault(register, {})[method] = selection
            setups[method] = (variant, val, test, base, int(selection["layer"]))

        fits = [(method, None, setups[method], 0.0, 0.0)
                for method in SOURCE_METHODS.values()]
        for variant, method in MIXED_METHODS.items():
            setup = setups[SOURCE_METHODS[variant]]
            budget, mass = float(selected[variant]["budget"]), float(selected[variant]["mass"])
            fits.extend((method, seed,
                         (setup[0], setup[1], setup[2],
                          _adapted_path(root, probe, register, variant), setup[4]),
                         budget, mass) for seed in SEEDS)
        yo = org.label.to_numpy()
        for method, seed, setup, budget, mass in fits:
            variant, val, test, base, layer = setup
            path = (base / f"layer_{layer:02d}.npz" if method in SOURCE_METHODS.values() else
                    base / f"seed{seed}_budget{budget:g}_organic{mass:g}.npz")
            Xv, Xs, Xo = (data.X(val, layer, "source"), data.X(test, layer, "source"),
                           data.X(org, layer, "organic"))
            yl, yt = val.label.to_numpy(), test.label.to_numpy()
            model = LinearProbe.load(path)
            sv, ss, so = model.score(Xv), model.score(Xs), model.score(Xo)
            threshold = macro_ba_threshold(val, sv)
            scale = float(np.std(sv))
            if scale <= 0:
                raise ValueError(f"constant validation scores: {probe}/{register}/{method}/{seed}")
            hist[(method, register)].append({
                "source_truth": (ss[yt == 0] - threshold) / scale,
                "source_hallucination": (ss[yt == 1] - threshold) / scale,
                "organic_truth": (so[yo == 0] - threshold) / scale,
                "organic_hallucination": (so[yo == 1] - threshold) / scale,
            })
            for dataset in DATASETS:
                vm = val.dataset.to_numpy() == dataset
                sm = test.dataset.to_numpy() == dataset
                om = org.dataset.to_numpy() == dataset
                stem = {"probe": probe, "dataset": dataset, "register": register,
                        "method": method, "seed": seed, "threshold": threshold, "layer": layer,
                        "source_variant": variant,
                        "budget": budget, "organic_mass": mass}
                organic_records.append({**stem, **full_metrics(yo[om], so[om], threshold)})
                source_records.append({**stem, **full_metrics(yt[sm], ss[sm], threshold)})
                thresholds.append({**stem, "source_validation_n": int(vm.sum()),
                                   "source_validation_balanced_accuracy":
                                       full_metrics(yl[vm], sv[vm], threshold)["balanced_accuracy"]})
                paired[(register, dataset, method, seed)] = {
                    "y": yo[om], "qids": org.question_id.to_numpy()[om],
                    "score": so[om], "threshold": threshold,
                }
    return (pd.DataFrame(organic_records), pd.DataFrame(source_records),
            pd.DataFrame(thresholds), hist, paired, selected, selections)


def paired_bootstrap(paired: dict, repeats: int = 200, seed: int = 42) -> pd.DataFrame:
    """Question-cluster intervals for both alternatives versus inorganic-only."""
    rng = np.random.RandomState(seed)

    def measures(y, score, threshold):
        pred = score >= threshold
        recall = float(np.mean(pred[y == 1])); fpr = float(np.mean(pred[y == 0]))
        return {"auroc": float(roc_auc_score(y, score)), "recall": recall, "fpr": fpr,
                "balanced_accuracy": (recall + 1 - fpr) / 2}

    def average(entries, idx):
        y = entries[0]["y"][idx]
        vals = [measures(y, x["score"][idx], x["threshold"]) for x in entries]
        return {key: float(np.mean([v[key] for v in vals])) for key in vals[0]}

    rows = []
    for register in REGISTERS:
        for alternative, alternative_seeds in (
                ("same_instruction", (None,)),
                ("inorganic_plus_organic", SEEDS),
                ("same_instruction_plus_organic", SEEDS)):
            blocks = []
            for dataset in DATASETS:
                base = [paired[(register, dataset, "inorganic_only", None)]]
                alternatives = [paired[(register, dataset, alternative, fit_seed)]
                                for fit_seed in alternative_seeds]
                for item in alternatives:
                    if (not np.array_equal(item["y"], base[0]["y"])
                            or not np.array_equal(item["qids"], base[0]["qids"])):
                        raise AssertionError("organic test predictions are not paired")
                unique, inverse = np.unique(base[0]["qids"], return_inverse=True)
                groups = [np.flatnonzero(inverse == i) for i in range(len(unique))]
                blocks.append((base, alternatives, groups))
            point = {key: [] for key in ("auroc", "recall", "fpr", "balanced_accuracy")}
            for base, alternatives, _ in blocks:
                idx = np.arange(len(base[0]["y"])); a, b = average(alternatives, idx), average(base, idx)
                for key in point: point[key].append(a[key] - b[key])
            draws = {key: [] for key in point}
            while len(draws["auroc"]) < repeats:
                diffs = {key: [] for key in point}
                for base, alternatives, groups in blocks:
                    idx = np.concatenate([groups[i] for i in rng.randint(len(groups), size=len(groups))])
                    if len(np.unique(base[0]["y"][idx])) != 2: break
                    a, b = average(alternatives, idx), average(base, idx)
                    for key in diffs: diffs[key].append(a[key] - b[key])
                else:
                    for key in draws: draws[key].append(float(np.mean(diffs[key])))
            for metric in point:
                low, high = np.percentile(draws[metric], [2.5, 97.5])
                rows.append({"register": register,
                             "comparison": f"{alternative} - inorganic_only",
                             "metric": metric, "difference": float(np.mean(point[metric])),
                             "ci95_low": float(low), "ci95_high": float(high),
                             "n_bootstrap": repeats, "unit": "question within dataset"})
    return pd.DataFrame(rows)


def _plt():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cross_probe_matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "savefig.facecolor": "white"})
    return plt


def plot_matrix(macro: pd.DataFrame, probe: str, out: Path) -> None:
    plt = _plt(); metrics = tuple(METRICS)
    labels = ("AUROC", "AUPRC", "Precision", "Recall", "FPR", "FNR", "Balanced acc.")
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2), sharey=True); cmap = plt.get_cmap("RdYlGn")
    for ax, register in zip(axes, REGISTERS):
        frame = macro.loc[macro.register == register].set_index("method").loc[list(METHODS)]
        values = frame[list(metrics)].to_numpy(); colors = values.copy(); colors[:, 4:6] = 1 - colors[:, 4:6]
        ax.imshow(colors, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(metrics)), labels, rotation=38, ha="left")
        ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=7)
        ax.set_yticks(range(len(METHODS)), [LABELS[m] for m in METHODS]); ax.tick_params(axis="y", length=0)
        ax.set_xticks(np.arange(-.5, len(metrics), 1), minor=True); ax.set_yticks(np.arange(-.5, len(METHODS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=3); ax.tick_params(which="minor", bottom=False, left=False)
        for i in range(len(METHODS)):
            for j in range(len(metrics)): ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=8.7, weight="bold")
        ax.set_title(register.title(), fontsize=13, weight="bold")
        for spine in ax.spines.values(): spine.set_visible(False)
    fig.suptitle(f"{PROBE_LABELS[probe]}: GPT-OSS transfer to organic answers", fontsize=16, y=.99)
    fig.text(.5, .035, "Equal-dataset mean; mixed fits average five seeds. Thresholds maximize balanced accuracy on inorganic validation data.", ha="center")
    fig.subplots_adjust(left=.15, right=.99, top=.78, bottom=.19, wspace=.14); fig.savefig(out, dpi=180); plt.close(fig)


def plot_confusions(counts: pd.DataFrame, probe: str, out: Path) -> None:
    plt = _plt(); fig, axes = plt.subplots(len(METHODS), 3, figsize=(12.2, 12.5))
    for row, method in enumerate(METHODS):
        for col, register in enumerate(REGISTERS):
            ax = axes[row, col]; r = counts.loc[(counts.method == method) & (counts.register == register)].iloc[0]
            cm = np.array([[r.tn, r.fp], [r.fn, r.tp]], dtype=float); ax.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2): ax.text(j, i, f"{cm[i,j]:.0f}", ha="center", va="center", weight="bold")
            if row == 0: ax.set_title(register.title(), weight="bold")
            if col == 0: ax.set_ylabel(f"{LABELS[method]}\nTrue label")
            if row == len(METHODS) - 1: ax.set_xlabel("Predicted label")
            ax.set_xticks([0, 1], ["Truth", "Hallucination"]); ax.set_yticks([0, 1], ["Truth", "Hallucination"])
    fig.suptitle(f"{PROBE_LABELS[probe]}: organic-test confusion matrices", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(out, dpi=180); plt.close(fig)


def plot_histograms(hist: dict, probe: str, out: Path) -> None:
    plt = _plt(); fig, axes = plt.subplots(len(METHODS), 3, figsize=(15, 12.5), sharey="row"); order = tuple(COLORS)
    for row, method in enumerate(METHODS):
        for col, register in enumerate(REGISTERS):
            ax = axes[row, col]; entries = hist[(method, register)]
            arrays = {key: np.concatenate([e[key] for e in entries]) for key in order}
            all_scores = np.concatenate(list(arrays.values())); lo, hi = np.quantile(all_scores, [.005, .995])
            bins = np.linspace(lo, hi, 45); repeat_weight = 1 / len(entries)
            for key in order:
                ax.hist(arrays[key], bins=bins, histtype="step", linewidth=1.8, color=COLORS[key],
                        weights=np.full(len(arrays[key]), repeat_weight), label=HIST_LABELS[key])
            ax.axvline(0, color="#25343D", linestyle="--", linewidth=1)
            if row == 0: ax.set_title(register.title(), weight="bold")
            if col == 0: ax.set_ylabel(f"{LABELS[method]}\nSamples")
            if row == len(METHODS) - 1: ax.set_xlabel("Probe score (threshold-centered, validation-SD units)")
    axes[0, 0].legend(fontsize=8); fig.suptitle(f"{PROBE_LABELS[probe]}: score distributions", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(out, dpi=180); plt.close(fig)


def plot_layers(root: Path, probe: str, selections: dict, out: Path) -> None:
    plt = _plt(); fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), sharey=True); rows = []
    styles = {"inorganic_only": ("basic", "#376A9A", "-"),
              "same_instruction": ("crossed", "#C66A2B", "--")}
    for ax, register in zip(axes, REGISTERS):
        notes = []
        for method, (variant, color, linestyle) in styles.items():
            grid = read_json(_source_path(root, probe, register, variant) / "grid.json")["layers"]
            layers = sorted(map(int, grid))
            values = [float(grid[str(i)]["validation_macro_auroc"]) for i in layers]
            selected = int(selections[register][method]["layer"])
            auc = float(grid[str(selected)]["validation_macro_auroc"])
            ax.plot(layers, values, color=color, linestyle=linestyle, linewidth=2,
                    label=LABELS[method])
            ax.scatter([selected], [auc], color=color, s=45, zorder=3)
            notes.append(f"{LABELS[method]}: L{selected}, {auc:.3f}")
            rows.extend({"probe": probe, "register": register, "method": method,
                         "source_variant": variant, "layer": layer,
                         "validation_macro_auroc": value, "selected": layer == selected}
                        for layer, value in zip(layers, values))
        ax.text(.04, .94, "\n".join(notes), transform=ax.transAxes, va="top", fontsize=8.5,
                bbox={"facecolor": "white", "edgecolor": "#D6DEE3", "alpha": .9})
        ax.set_title(register.title(), weight="bold"); ax.set_xlabel("Layer (0 = embeddings)"); ax.grid(alpha=.25)
    axes[0].set_ylabel("Inorganic-validation AUROC"); axes[-1].legend(fontsize=8)
    fig.suptitle(f"{PROBE_LABELS[probe]}: independent source-layer selection", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(out, dpi=180); plt.close(fig)
    pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False)


def report(data: PooledData, root: Path, out: Path, probes: tuple[str, ...]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "manifest.json", {"model": data.model, "datasets": DATASETS, "registers": REGISTERS,
               "methods": METHODS, "source_variants": SOURCE_VARIANTS,
               "threshold": "equal-dataset inorganic-validation balanced accuracy", "seeds": SEEDS,
               "excluded_cross_dataset_collisions": data.collisions})
    counts = []
    for dataset in DATASETS:
        for split in ("train", "validation", "test"):
            org = data.organic.loc[(data.organic.dataset == dataset) & (data.organic.split == split)]
            counts.append({"dataset": dataset, "split": split, "domain": "organic", "register": "all",
                           "questions": org.question_id.nunique(), "answers": len(org),
                           "truth": int((org.label == 0).sum()), "hallucination": int((org.label == 1).sum())})
            for register in REGISTERS:
                for variant in SOURCE_VARIANTS:
                    src = data.source_for(register, variant)
                    part = src.loc[(src.dataset == dataset) & (src.split == split)]
                    counts.append({"dataset": dataset, "split": split, "domain": "inorganic",
                                   "source_variant": variant, "register": register,
                                   "questions": part.question_id.nunique(), "answers": len(part),
                                   "truth": int((part.label == 0).sum()),
                                   "hallucination": int((part.label == 1).sum())})
    pd.DataFrame(counts).to_csv(out / "sample_counts.csv", index=False)
    overview = []
    for probe in probes:
        org, src, thresholds, hist, paired, selected, selections = collect(data, root, probe)
        target = out / probe; target.mkdir(parents=True, exist_ok=True)
        org_macro, src_macro = macro_table(org), macro_table(src); cms = pooled_confusions(org)
        org.to_csv(target / "organic_test_per_dataset.csv", index=False); src.to_csv(target / "inorganic_test_per_dataset.csv", index=False)
        org_macro.to_csv(target / "organic_test_macro.csv", index=False); src_macro.to_csv(target / "inorganic_test_macro.csv", index=False)
        thresholds.to_csv(target / "validation_thresholds.csv", index=False); cms.to_csv(target / "organic_test_confusion_counts.csv", index=False)
        paired_bootstrap(paired).to_csv(target / "paired_transfer_deltas_ci95.csv", index=False)
        overall = org_macro.groupby(["probe", "method"], as_index=False)[list(METRICS)].mean()
        overall.to_csv(target / "organic_test_overall.csv", index=False); overview.append(overall)
        training = []
        for variant, mix in selected.items():
            for seed in SEEDS:
                chosen = organic_question_sample(data.organic, float(mix["budget"]), seed)
                for dataset in DATASETS:
                    part = chosen.loc[chosen.dataset == dataset]
                    training.append({"probe": probe, "source_variant": variant,
                                     "organic_mass": float(mix["mass"]), "seed": seed,
                                     "dataset": dataset,
                                     "organic_questions": part.question_id.nunique(),
                                     "organic_answers": len(part),
                                     "organic_truth": int((part.label == 0).sum()),
                                     "organic_hallucination": int((part.label == 1).sum())})
        pd.DataFrame(training).to_csv(target / "selected_training_counts.csv", index=False)
        write_json(target / "selected_mix.json", {"mixes": selected, "layer_selections": selections})
        grids = []
        for variant in SOURCE_VARIANTS:
            suffix = "" if variant == SOURCE_VARIANT else f"_{variant}"
            grid = pd.read_csv(root / "search" / probe / f"mix_grid{suffix}.csv")
            grid.insert(0, "source_variant", variant); grids.append(grid)
        pd.concat(grids, ignore_index=True).to_csv(target / "mix_validation_grid.csv", index=False)
        plot_matrix(org_macro, probe, target / "organic_test_comparison.png")
        plot_confusions(cms, probe, target / "organic_test_confusions.png")
        plot_histograms(hist, probe, target / "probe_score_histograms.png")
        plot_layers(root, probe, selections, target / "layer_selection_auroc.png")
        print(f"reported {probe} -> {target}", flush=True)
    completed = [pd.read_csv(path) for path in sorted(out.glob("*/organic_test_overall.csv"))]
    pd.concat(completed, ignore_index=True).to_csv(out / "transfer_overview.csv", index=False)

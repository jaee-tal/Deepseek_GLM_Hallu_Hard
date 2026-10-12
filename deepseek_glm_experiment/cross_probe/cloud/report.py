"""Human-readable tables and figures for the four-arm cloud experiment."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..transfer_metrics import METRICS, PROBE_LABELS
from ..vendor.io_utils import read_json, write_json
from .probes import ARMS, binary_auc

ARM_LABELS = {
    "plain_inorganic": "Inorganic",
    "same_instruction": "Same instruction",
    "plain_inorganic_plus_organic": "Inorganic + 20% organic",
    "same_instruction_plus_organic": "Same instruction + 20% organic",
}
REGISTER_LABELS = {"lie": "Lie", "misremember": "Misremember", "unknown": "Unknown"}
COLORS = {"source_truth": "#2166ac", "source_hallucination": "#b2182b",
          "organic_truth": "#1b9e77", "organic_hallucination": "#f28e2b"}


def _result_dirs(root: Path):
    for path in sorted((root / "models").glob("*/*/*/*/result.json")):
        yield path.parent, read_json(path)


def _flat_results(root: Path) -> pd.DataFrame:
    rows = []
    for _, result in _result_dirs(root):
        common = {k: result[k] for k in ("model", "probe", "register", "arm", "seed", "layer", "param",
                                          "selection_auroc", "threshold", "n_organic_questions",
                                          "n_organic_rows", "n_source_train_rows")}
        for distribution in ("source_test", "organic_test"):
            macro = result["metrics"][distribution]["equal_dataset_macro"]
            rows.append({**common, "distribution": distribution, **macro,
                         "auroc_ci_low": (result["metrics"].get("organic_test_auroc_ci95", [None, None])[0]
                                          if distribution == "organic_test" else None),
                         "auroc_ci_high": (result["metrics"].get("organic_test_auroc_ci95", [None, None])[1]
                                           if distribution == "organic_test" else None)})
    return pd.DataFrame(rows)


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = [m for m in METRICS if m in frame] + ["layer", "selection_auroc", "threshold",
                                                     "n_organic_questions", "n_organic_rows", "n_source_train_rows"]
    return frame.groupby(["model", "probe", "register", "arm", "distribution"], as_index=False)[numeric].mean()


def _heatmap(ax, values, row_labels, col_labels, title, fmt=".3f", cmap="viridis", vmin=0, vmax=1):
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=8)
    ax.set_title(title, fontsize=10)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            ax.text(j, i, format(value, fmt), ha="center", va="center",
                    color="white" if value < .25 or value > .72 else "black", fontsize=8)
    return im


def metric_matrices(frame: pd.DataFrame, out: Path, probe: str) -> None:
    organic = frame[(frame.probe == probe) & (frame.distribution == "organic_test")]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    for ax, metric in zip(axes.flat, METRICS):
        values = np.array([[organic.loc[(organic.register == register) & (organic.arm == arm), metric].iloc[0]
                            for arm in ARMS] for register in REGISTER_LABELS])
        _heatmap(ax, values, list(REGISTER_LABELS.values()), [ARM_LABELS[a] for a in ARMS], metric.upper())
    axes.flat[-1].axis("off")
    fig.suptitle(f"{PROBE_LABELS.get(probe, probe)} — organic test (equal-dataset macro)", fontsize=14)
    fig.savefig(out / "organic_transfer_metric_matrices.png", dpi=180)
    plt.close(fig)


def confusion_figure(root: Path, out: Path, probe: str) -> None:
    collected: dict[tuple[str, str], list[np.ndarray]] = {}
    for _, result in _result_dirs(root):
        if result["probe"] != probe:
            continue
        total = np.zeros((2, 2), dtype=float)
        for metric in result["metrics"]["organic_test"]["by_dataset"].values():
            total += np.array([[metric["tn"], metric["fp"]], [metric["fn"], metric["tp"]]])
        collected.setdefault((result["register"], result["arm"]), []).append(total)
    fig, axes = plt.subplots(len(REGISTER_LABELS), len(ARMS), figsize=(15, 11), constrained_layout=True)
    for i, register in enumerate(REGISTER_LABELS):
        for j, arm in enumerate(ARMS):
            ax = axes[i, j]
            matrix = np.mean(collected[(register, arm)], axis=0)
            normalized = matrix / matrix.sum(axis=1, keepdims=True)
            ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
            for y in range(2):
                for x in range(2):
                    ax.text(x, y, f"{matrix[y, x]:.1f}\n{normalized[y, x]:.1%}", ha="center", va="center", fontsize=8)
            ax.set_xticks([0, 1], ["Pred truth", "Pred halluc."])
            ax.set_yticks([0, 1], ["Real truth", "Real halluc."])
            ax.set_title(f"{REGISTER_LABELS[register]} · {ARM_LABELS[arm]}", fontsize=9)
    fig.suptitle(f"{PROBE_LABELS.get(probe, probe)} — organic test confusion matrices", fontsize=14)
    fig.savefig(out / "organic_test_confusion_matrices.png", dpi=180)
    plt.close(fig)


def _mean_scores(entries: list[tuple[Path, dict]], prefix: str):
    arrays = [np.load(path / "scores.npz", allow_pickle=False) for path, _ in entries]
    labels = arrays[0][f"{prefix}_label"]
    scores = np.mean([a[f"{prefix}_score"] for a in arrays], axis=0)
    return labels, scores


def score_histograms(root: Path, out: Path, probe: str) -> None:
    entries = [(path, result) for path, result in _result_dirs(root) if result["probe"] == probe]
    hist_dir = out / "score_histograms"
    hist_dir.mkdir(exist_ok=True)
    for register in REGISTER_LABELS:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        for ax, arm in zip(axes.flat, ARMS):
            chosen = [(path, result) for path, result in entries
                      if result["register"] == register and result["arm"] == arm]
            sy, ss = _mean_scores(chosen, "source")
            oy, oscore = _mean_scores(chosen, "organic")
            all_scores = np.concatenate((ss, oscore))
            lo, hi = np.quantile(all_scores, [.005, .995])
            bins = np.linspace(lo, hi, 50) if hi > lo else 20
            for scores, labels, label, color in (
                (ss, sy, "Inorganic truth", COLORS["source_truth"]),
                (ss, sy, "Inorganic hallucination", COLORS["source_hallucination"]),
                (oscore, oy, "Organic truth", COLORS["organic_truth"]),
                (oscore, oy, "Organic hallucination", COLORS["organic_hallucination"]),
            ):
                wanted = 1 if "hallucination" in label.lower() else 0
                ax.hist(scores[labels == wanted], bins=bins, histtype="step", linewidth=1.7,
                        density=True, label=label, color=color)
            ax.axvline(np.mean([r["threshold"] for _, r in chosen]), color="black", linestyle="--", linewidth=1)
            ax.set_title(ARM_LABELS[arm])
            ax.set_xlabel("Probe score")
            ax.set_ylabel("Normalized sample density")
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f"{PROBE_LABELS.get(probe, probe)} · {REGISTER_LABELS[register]}", fontsize=14)
        fig.savefig(hist_dir / f"{register}_activation_score_distributions.png", dpi=180)
        plt.close(fig)


def layer_curves(root: Path, out: Path, probe: str) -> None:
    result_entries = [(path, result) for path, result in _result_dirs(root) if result["probe"] == probe]
    model_name = result_entries[0][1]["model"] if result_entries else ""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for ax, register in zip(axes, REGISTER_LABELS):
        for arm in ARMS:
            arm_results = [result for _, result in result_entries
                           if result["register"] == register and result["arm"] == arm]
            test_auc = np.mean([result["metrics"]["organic_test"]["equal_dataset_macro"]["auroc"]
                                for result in arm_results])
            if not ARMS[arm]["organic"]:
                grids = []
                for path in sorted((root / "models" / probe / register / arm).glob("*/grid.json")):
                    grid = read_json(path)["layers"]
                    ordered = sorted(map(int, grid))
                    grids.append([grid[str(i)]["selection_auroc"] for i in ordered])
                curve = np.mean(grids, axis=0)
                best = int(arm_results[0]["layer"])
                ax.plot(ordered, curve, label=ARM_LABELS[arm], linewidth=1.5)
                best_y = float(np.mean([r["selection_auroc"] for r in arm_results]))
                ax.scatter([best], [best_y], s=25)
                ax.annotate(f"L{best}: source val {best_y:.3f}\ntest {test_auc:.3f}",
                            (best, best_y), fontsize=7)
            else:
                inherited_layers = {int(result["layer"]) for result in arm_results}
                if len(inherited_layers) != 1:
                    raise ValueError(f"mixed arm {probe}/{register}/{arm} did not inherit one source layer")
                layer = inherited_layers.pop()
                val_auc = float(np.mean([result["selection_auroc"] for result in arm_results]))
                ax.scatter([layer], [val_auc], marker="X", s=55, label=ARM_LABELS[arm])
                ax.annotate(f"L{layer} inherited\norg val {val_auc:.3f}; test {test_auc:.3f}",
                            (layer, val_auc), fontsize=7)
        if model_name == "deepseek_v41_flash":
            ax.axvline(20.5, color="grey", linestyle=":", linewidth=1)
            ax.text(20.5, .02, "encoder / decoder", rotation=90, fontsize=7, color="grey", va="bottom")
        ax.set_title(REGISTER_LABELS[register])
        ax.set_xlabel("Residual-stream layer (0 = embeddings)")
        ax.set_ylabel("Validation AUROC")
        ax.set_ylim(0, 1)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{PROBE_LABELS.get(probe, probe)} — source-selected layers reused by mixed arms", fontsize=14)
    fig.savefig(out / "layer_selection_curves.png", dpi=180)
    plt.close(fig)


def _paired_delta(path_a: Path, path_b: Path, repeats: int = 1000, seed: int = 42):
    a, b = np.load(path_a / "scores.npz", allow_pickle=False), np.load(path_b / "scores.npz", allow_pickle=False)
    for key in ("organic_label", "organic_dataset", "organic_question_id"):
        if not np.array_equal(a[key], b[key]):
            raise ValueError(f"paired score rows differ for {path_a} and {path_b}: {key}")
    y, dataset, qid = a["organic_label"], a["organic_dataset"], a["organic_question_id"]
    sa, sb = a["organic_score"], b["organic_score"]
    point = np.mean([binary_auc(y[dataset == d], sa[dataset == d]) -
                     binary_auc(y[dataset == d], sb[dataset == d]) for d in np.unique(dataset)])
    rng, values = np.random.RandomState(seed), []
    groups = {(d, q): np.flatnonzero((dataset == d) & (qid == q)) for d in np.unique(dataset)
              for q in np.unique(qid[dataset == d])}
    questions = {d: np.unique(qid[dataset == d]) for d in np.unique(dataset)}
    for _ in range(repeats):
        delta = []
        for d, qs in questions.items():
            draw = rng.choice(qs, size=len(qs), replace=True)
            idx = np.concatenate([groups[(d, q)] for q in draw])
            if len(np.unique(y[idx])) != 2:
                break
            delta.append(binary_auc(y[idx], sa[idx]) - binary_auc(y[idx], sb[idx]))
        if len(delta) == len(questions):
            values.append(np.mean(delta))
    ci = np.percentile(values, [2.5, 97.5]) if values else [np.nan, np.nan]
    return float(point), float(ci[0]), float(ci[1])


def paired_deltas(root: Path) -> pd.DataFrame:
    entries = list(_result_dirs(root))
    rows = []
    probes = sorted({result["probe"] for _, result in entries})
    registers = sorted({result["register"] for _, result in entries})
    for probe in probes:
        for register in registers:
            baseline = next(path for path, result in entries if result["probe"] == probe
                            and result["register"] == register and result["arm"] == "plain_inorganic")
            for path, result in entries:
                if result["probe"] != probe or result["register"] != register or result["arm"] == "plain_inorganic":
                    continue
                delta, low, high = _paired_delta(path, baseline, seed=42 + (result.get("seed") or 0))
                rows.append({"probe": probe, "register": register, "arm": result["arm"],
                             "seed": result.get("seed"), "delta_auroc_vs_plain_inorganic": delta,
                             "ci95_low": low, "ci95_high": high})
    return pd.DataFrame(rows)


def report(root: Path, output: Path, probes: tuple[str, ...]) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    trials = _flat_results(root)
    if trials.empty:
        raise RuntimeError(f"no completed cloud probe results under {root}")
    summary = _aggregate(trials)
    trials.to_csv(output / "all_trials.csv", index=False)
    summary.to_csv(output / "equal_dataset_summary.csv", index=False)
    counts = trials[["model", "probe", "register", "arm", "seed", "n_source_train_rows",
                     "n_organic_questions", "n_organic_rows"]].drop_duplicates()
    counts.to_csv(output / "training_sample_counts.csv", index=False)
    detailed = []
    for _, result in _result_dirs(root):
        for distribution, by_dataset in result.get("sample_counts", {}).items():
            for dataset, values in by_dataset.items():
                detailed.append({"model": result["model"], "probe": result["probe"],
                                 "register": result["register"], "arm": result["arm"],
                                 "seed": result.get("seed"), "distribution": distribution,
                                 "dataset": dataset, **values})
    pd.DataFrame(detailed).to_csv(output / "sample_counts_by_dataset_and_label.csv", index=False)
    paired_deltas(root).to_csv(output / "paired_organic_auroc_deltas.csv", index=False)
    for probe in probes:
        probe_out = output / probe
        probe_out.mkdir(exist_ok=True)
        subset = summary[summary.probe == probe]
        subset.to_csv(probe_out / "metrics.csv", index=False)
        metric_matrices(summary, probe_out, probe)
        confusion_figure(root, probe_out, probe)
        score_histograms(root, probe_out, probe)
        layer_curves(root, probe_out, probe)
    manifest = {"model": trials.model.iloc[0], "probes": list(probes), "arms": list(ARMS),
                "registers": list(REGISTER_LABELS), "n_trials": int(len(trials) // 2),
                "primary_result": "equal_dataset_summary.csv"}
    write_json(output / "report_manifest.json", manifest)
    return summary

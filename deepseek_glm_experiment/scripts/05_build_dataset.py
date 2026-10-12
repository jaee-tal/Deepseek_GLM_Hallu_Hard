"""Join everything into dataset/rows.parquet, write layer-major activations, funnel and audit examples."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from _common import base_parser, names, run_dir_for, setup

from cross_probe.config import questions_path, split_path
from cross_probe.dataset import ROW_COLS, build_rows, write_layer_major
from cross_probe.vendor import provenance
from cross_probe.vendor.io_utils import read_json, write_json

log = logging.getLogger("build_dataset")


def audit_markdown(df: pd.DataFrame, n_per_state: int = 10, seed: int = 0) -> str:
    rng = np.random.RandomState(seed)
    lines = ["# Audit examples (inspect before training)", ""]
    for st in ("O0", "O1", "I0", "I1"):
        d = df[df["state"] == st]
        lines.append(f"## {st} (n={len(d)})")
        for i in rng.choice(len(d), size=min(n_per_state, len(d)), replace=False) if len(d) else []:
            r = d.iloc[int(i)]
            lines.append(f"- [{r['split']}, {r['condition']}, t{r['template_id']}] **{r['question_id']}** → `{r['response_text'][:120]}` (verdict {r['verdict']} via {r['verdict_source']})")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = base_parser("build dataset")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    cfg = setup(args)
    _, dataset = names(cfg, args)
    run_dir = run_dir_for(cfg, args)
    d = run_dir / "dataset"
    d.mkdir(exist_ok=True)
    verd = run_dir / "judge" / "verdicts.jsonl"
    if (d / "acts_layer_major.npy").exists() and (d / "rows.parquet").exists() and (d / "rows.parquet").stat().st_mtime > verd.stat().st_mtime and not args.force:
        print(f"dataset already built after the last judge update: {d} (use --force to rebuild)")
        return
    known_path = run_dir / "known_questions.json"
    if not known_path.exists():
        known_path = cfg.data_dir(dataset) / f"known_{cfg.model_name(args.model)}.json"
    known = set(read_json(known_path)["question_ids"]) if known_path.exists() else None
    if known is None:
        log.warning("no %s — instructed rows will NOT be restricted to known questions", known_path.name)
    else:
        log.info("restricting instructed rows to %d known questions", len(known))
    kept, funnel = build_rows(run_dir, questions_path(cfg, dataset), split_path(cfg, dataset), known)
    funnel["known_questions"] = (len(known) if known is not None else None)
    kept["ds_row"] = np.arange(len(kept))
    kept[ROW_COLS + ["ds_row", "alignment_ok", "is_duplicate"]].to_parquet(d / "rows.parquet", index=False)
    write_json(d / "funnel.json", funnel)
    write_layer_major(run_dir / "replay" / "activations.npy", kept["row"].to_numpy(), d / "acts_layer_major.npy", log)
    (run_dir / "results").mkdir(exist_ok=True)
    (run_dir / "results" / "audit_examples.md").write_text(audit_markdown(kept), encoding="utf-8")
    # descriptive: organic error rate by popularity decile (all splits; labels only, no probe involvement)
    org = kept[kept["domain"] == "organic"]
    has_dec = len(org) and org["s_pop_decile"].notna().any()
    by_dec = org.dropna(subset=["s_pop_decile"]).groupby("s_pop_decile")["label"].mean().to_dict() if has_dec else {}
    write_json(d / "descriptives.json", {"organic_error_by_decile": {str(k): float(v) for k, v in by_dec.items()}})
    provenance.write(run_dir, {"funnel": funnel}, section="dataset")
    print("funnel:", funnel)
    print(f"audit -> {run_dir / 'results' / 'audit_examples.md'}")


if __name__ == "__main__":
    main()

"""Sample this dataset's questions and assign the 60/20/20 question split.

Extending the sample is additive and non-destructive: the enlarged sample must contain every
question already sampled (the seeded draw is a superset by construction), and questions already
assigned to a split keep that assignment. Responses, probe fits and published cells are tied to
those ids and splits, so anything else would silently invalidate work already done.
"""
from __future__ import annotations

from _common import base_parser, names, setup

from cross_probe.config import questions_path, split_path
from cross_probe.data import build_split, load_questions
from cross_probe.vendor.io_utils import read_json, read_jsonl, sha256_file, write_json, write_jsonl
from cross_probe.vendor.modeling import setup_hf_env


def main() -> None:
    p = base_parser("prepare questions + split")
    p.add_argument("--force", action="store_true", help="allow a sample that is not a superset of the existing one")
    args = p.parse_args()
    cfg = setup(args)
    setup_hf_env(cfg.hf_home())
    _, dataset = names(cfg, args)
    qp, sp = questions_path(cfg, dataset), split_path(cfg, dataset)
    qp.parent.mkdir(parents=True, exist_ok=True)

    prior_q = read_jsonl(qp) if qp.exists() else []
    prior_split = read_json(sp) if sp.exists() else None
    prior_ids = {q["question_id"] for q in prior_q}

    qs, info = load_questions(cfg, dataset)
    ids = {q["question_id"] for q in qs}

    split_covers = bool(prior_split) and ids <= {q for n in ("train", "validation", "test") for q in prior_split.get(n, [])}
    if prior_ids and prior_ids == ids and split_covers:
        print(f"unchanged: {len(ids)} questions in {qp} ({sha256_file(qp)[:16]})")
        return
    if prior_ids == ids and not split_covers:
        print("questions unchanged but the split does not cover all of them — rebuilding the split")
    dropped = prior_ids - ids
    if dropped and not args.force:
        raise SystemExit(f"the new sample drops {len(dropped)} question(s) already sampled — responses and probe fits "
                         f"are tied to them. Re-run with --force only if you intend to discard that work.")
    split = build_split(qs, cfg, dataset, existing=prior_split)
    write_jsonl(qp, qs)
    s = cfg["split"]
    write_json(sp, {"seed": int(s["seed"]), "unit": "question_id", "n_questions": len(qs), "sampling": info, **split})
    added = len(ids - prior_ids)
    print(f"[{dataset}] {len(qs)} questions ({added} new, {len(prior_ids)} kept) -> {qp}")
    print(f"  split {[len(split[k]) for k in ('train', 'validation', 'test')]} -> {sp}")
    print("  sampling:", {k: v for k, v in info.items() if k != "bins"})


if __name__ == "__main__":
    main()

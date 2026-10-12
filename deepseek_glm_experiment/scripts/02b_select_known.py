"""Select the questions the model demonstrably knows, from its organic answers alone.

Inorganic rounds are built only from these. Without the restriction, I0 (truthful, correct) exists
only where the model knows the answer while I1 (instructed error) survives mostly where it does not
— so the two classes come from near-disjoint question populations and a probe can separate them on
knowledge rather than on truthfulness.

Writes data/<dataset>/known_<model>.json and reports the known rate, which decides how many samples
per question the inorganic rounds need.
"""
from __future__ import annotations

import collections

from _common import base_parser, names, run_dir_for, setup

from cross_probe.config import questions_path, split_path
from cross_probe.jobs import error_samples_for
from cross_probe.vendor.io_utils import read_json, read_jsonl, write_json
from cross_probe.vendor.splits import split_lookup

CRITERIA = ("ever_correct", "never_wrong", "correct_never_wrong")


def eligible_questions(by_q: dict[str, list[str]], criterion: str) -> set[str]:
    """The strict criterion excludes any question with an organic factual error."""
    if criterion not in CRITERIA:
        raise ValueError(criterion)
    if criterion == "ever_correct":
        return {q for q, vs in by_q.items() if "correct" in vs}
    if criterion == "never_wrong":
        return {q for q, vs in by_q.items() if vs and "incorrect" not in vs}
    return {q for q, vs in by_q.items() if "correct" in vs and "incorrect" not in vs}


def main() -> None:
    p = base_parser("select the questions the model answers correctly organically")
    p.add_argument("--criterion", default="ever_correct", choices=list(CRITERIA),
                   help="ever_correct: >=1 correct; never_wrong: no incorrect; "
                        "correct_never_wrong: >=1 correct and no incorrect (strict GPT-OSS gate).")
    p.add_argument("--projected-obey", type=float, default=0.20,
                   help="obey rate assumed only for the projected-yield message (PopQA measured 0.20 on known questions)")
    args = p.parse_args()
    cfg = setup(args)
    model_name, dataset = names(cfg, args)
    run_dir = run_dir_for(cfg, args)

    verd = {v["conversation_id"]: v["verdict"] for v in read_jsonl(run_dir / "judge" / "verdicts.jsonl")}
    by_q: dict[str, list[str]] = collections.defaultdict(list)
    invalid_organic_qids: set[str] = set()
    for g in read_jsonl(run_dir / "generations.jsonl"):
        if g["condition"] != "organic":
            continue
        if args.criterion == "correct_never_wrong" and (g.get("exclude_reason") or g.get("truncated")):
            invalid_organic_qids.add(g["question_id"])
        v = verd.get(g["conversation_id"])
        if v in ("correct", "incorrect", "abstain"):
            by_q[g["question_id"]].append(v)
    if not by_q:
        raise SystemExit("no judged organic responses found — run 02_generate --conditions organic and 04_judge first")

    if args.criterion == "correct_never_wrong":
        expected = int(cfg.generation(model_name, dataset)["organic_samples_per_question"])
        planned = run_dir / "questions_used.json"
        expected_qids = set(read_json(planned)["question_ids"]) if planned.exists() else set(by_q)
        generated_qids = {g["question_id"] for g in read_jsonl(run_dir / "generations.jsonl") if g["condition"] == "organic"}
        incomplete = sorted(q for q in expected_qids if len(by_q.get(q, [])) != expected)
        if generated_qids != expected_qids or incomplete:
            raise SystemExit(f"strict eligibility needs {expected} judged organic answers per planned question; "
                             f"missing generation for {len(expected_qids - generated_qids)} questions, "
                             f"incomplete judgments for {len(incomplete)} questions")
    ever_correct = eligible_questions(by_q, "ever_correct")
    never_wrong = eligible_questions(by_q, "never_wrong")
    chosen = eligible_questions(by_q, args.criterion)
    if args.criterion == "correct_never_wrong":
        chosen -= invalid_organic_qids
    if args.criterion == "correct_never_wrong" and not chosen:
        raise SystemExit("no eligible questions: cannot generate the GPT-OSS inorganic conditions")
    n_all = len(by_q)

    n_q_total = len(read_jsonl(questions_path(cfg, dataset)))
    split = read_json(split_path(cfg, dataset))
    lookup = split_lookup({s: split[s] for s in ("train", "validation", "test")})
    counts_by_split = {
        s: {"questions_with_organic_verdicts": sum(lookup[q] == s for q in by_q),
            "eligible_questions": sum(lookup[q] == s for q in chosen),
            "questions_with_any_organic_error": sum(lookup[q] == s and "incorrect" in vs for q, vs in by_q.items())}
        for s in ("train", "validation", "test")}
    out = {"model": model_name, "dataset": dataset, "criterion": args.criterion,
           "n_questions_total": n_q_total, "n_questions_with_organic_verdicts": n_all,
           "n_known": len(chosen), "known_rate": (len(chosen) / n_all if n_all else None),
           "counts": {"ever_correct": len(ever_correct), "never_wrong": len(never_wrong),
                      "ever_wrong": len({q for q, vs in by_q.items() if "incorrect" in vs}),
                      "never_correct": n_all - len(ever_correct),
                      "invalid_organic_generation_questions": len(invalid_organic_qids)},
           "counts_by_split": counts_by_split,
           "question_ids": sorted(chosen)}
    path = cfg.data_dir(dataset) / f"known_{model_name}.json"
    write_json(path, out)
    write_json(run_dir / "known_questions.json", out)

    n_err = error_samples_for(len(chosen), int(cfg.generation(model_name, dataset).get("error_rows_target", 8000)))
    projected = len(chosen) * n_err * args.projected_obey
    print(f"[{model_name}/{dataset}] organic verdicts for {n_all} of {n_q_total} questions")
    for k, v in out["counts"].items():
        print(f"    {k:16s} {v:6d}  ({v / n_all:.1%})")
    print(f"  criterion {args.criterion!r} -> {len(chosen)} known questions ({out['known_rate']:.1%}) -> {path}")
    print(f"  adaptive sampling: {n_err} error samples/question/register")
    print(f"  projected usable I1 rows per register at an obey rate of {args.projected_obey:.0%}: ~{projected:.0f}")
    if projected < 1000:
        print("  WARNING: projected yield is below ~1000 rows per register — this dataset may be underpowered "
              "under the knowledge restriction. Consider enlarging the question sample before generating.")


if __name__ == "__main__":
    main()

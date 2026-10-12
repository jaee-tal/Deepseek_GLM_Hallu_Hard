"""Grade every response (alias rule + GPT-mini arbiter, or reference-grounded GPT-mini). No GPU."""
from __future__ import annotations

from _common import base_parser, gate, names, run_dir_for, setup

from cross_probe.config import questions_path
from cross_probe.judge import check_credit, judge_all
from cross_probe.vendor import provenance
from cross_probe.vendor.io_utils import read_json, read_jsonl


def main() -> None:
    p = base_parser("judge responses")
    p.add_argument("--limit-llm", type=int, default=None, help="cap on newly judged rows (debug)")
    args = p.parse_args()
    cfg = setup(args)
    _, dataset = names(cfg, args)
    dcfg = cfg.dataset(dataset)
    run_dir = run_dir_for(cfg, args)
    gate(cfg, args, require_gpu=False)
    credit = check_credit()
    records = read_jsonl(run_dir / "generations.jsonl")
    questions = {q["question_id"]: q for q in read_jsonl(questions_path(cfg, dataset))}
    jcfg = dict(cfg["judge"])
    if dcfg.get("reference_max_words"):
        jcfg["reference_max_words"] = int(dcfg["reference_max_words"])
    verdicts = judge_all(records, questions, jcfg, run_dir / "judge", limit_llm=args.limit_llm, mode=dcfg["judge_mode"])
    stats = read_json(run_dir / "judge" / "stats.json")
    provenance.write(run_dir, {"config": jcfg, "mode": dcfg["judge_mode"], "stats": stats, "credit_preflight": credit}, section="judge")
    print(f"{len(verdicts)} verdicts; stats: {stats}")


if __name__ == "__main__":
    main()

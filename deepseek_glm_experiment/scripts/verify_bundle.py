#!/usr/bin/env python3
"""Offline structural verification for the portable DeepSeek/GLM bundle."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cross_probe.config import load_config  # noqa: E402
from cross_probe.vendor.io_utils import read_json, read_jsonl, sha256_file  # noqa: E402


def main() -> None:
    cfg = load_config(ROOT / "configs" / "cloud_experiment.yaml")
    expected_models = {"deepseek_v41_flash", "glm_53_flash"}
    configured_models = set(cfg["cloud"]["models"])
    if configured_models != expected_models:
        raise SystemExit(f"model registry mismatch: {configured_models}")

    expected_counts = {
        "pop_qa": (9000, (5400, 1800, 1800)),
        "trivia_qa": (5000, (3000, 1000, 1000)),
        "hallu_hard": (700, (420, 140, 140)),
    }
    report = {"config_hash": cfg.hash, "models": sorted(configured_models), "inputs": {}}
    for dataset, (n_expected, split_expected) in expected_counts.items():
        qpath = ROOT / "data" / dataset / "questions.jsonl"
        spath = ROOT / "data" / dataset / "split_seed42.json"
        questions = read_jsonl(qpath)
        split = read_json(spath)
        counts = tuple(len(split[name]) for name in ("train", "validation", "test"))
        ids = [qid for name in ("train", "validation", "test") for qid in split[name]]
        qids = {q["question_id"] for q in questions}
        if len(questions) != n_expected or counts != split_expected:
            raise SystemExit(f"{dataset}: expected {n_expected}/{split_expected}, got {len(questions)}/{counts}")
        if len(ids) != len(set(ids)) or set(ids) != qids:
            raise SystemExit(f"{dataset}: split does not cover every question exactly once")
        item = {
            "questions": len(questions),
            "split": dict(zip(("train", "validation", "test"), counts)),
            "questions_sha256": sha256_file(qpath),
            "split_sha256": sha256_file(spath),
        }
        if dataset == "hallu_hard":
            domains = Counter(q["meta"]["domain"] for q in questions)
            expected_domains = {"medical_guidelines": 250, "research_questions": 250,
                                "legal_cases": 100, "coding": 100}
            if dict(domains) != expected_domains:
                raise SystemExit(f"HalluHard domain counts mismatch: {dict(domains)}")
            item["domains"] = dict(domains)
        report["inputs"][dataset] = item

    for path in (ROOT / "runs", ROOT / "results", ROOT / "cloud_artifacts", ROOT / "wandb", ROOT / "logs"):
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()

    print(json.dumps(report, indent=2))
    print("Bundle verification passed.")


if __name__ == "__main__":
    main()

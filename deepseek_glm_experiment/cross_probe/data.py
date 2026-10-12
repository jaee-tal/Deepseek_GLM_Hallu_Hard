"""Question loading for every dataset, sampling (no correctness information is ever used) and the question split.

All loaders return the same record schema:
    {"question_id", "question", "native_id", "meta": {"answer_value", "aliases", "reference"?, "domain"?, ...}}
plus, for PopQA only, a top-level "s_pop_decile".
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config


# ----------------------------------------------------------------------------- record mapping
def popqa_row_to_record(r: dict[str, Any], id_prefix: str) -> dict[str, Any]:
    aliases = json.loads(r["possible_answers"]) if isinstance(r["possible_answers"], str) else list(r["possible_answers"])
    aliases = [str(a) for a in aliases]
    if r["obj"] not in aliases:
        aliases = [r["obj"]] + aliases
    return {"question_id": f"{id_prefix}_{r['id']}", "question": r["question"].strip(), "native_id": str(r["id"]),
            "meta": {"answer_value": r["obj"], "aliases": aliases, "prop": r.get("prop"), "subj": r.get("subj"),
                     "s_pop": int(r.get("s_pop") or 0), "o_pop": int(r.get("o_pop") or 0),
                     "s_wiki_title": r.get("s_wiki_title"), "o_wiki_title": r.get("o_wiki_title")}}


def trivia_row_to_record(r: dict[str, Any], id_prefix: str) -> dict[str, Any]:
    a = r["answer"]
    value = str(a["value"])
    aliases = [str(x) for x in (a.get("aliases") or [])]
    if value not in aliases:
        aliases = [value] + aliases
    nid = str(r.get("question_id") or r.get("id"))
    return {"question_id": f"{id_prefix}_{nid}", "question": str(r["question"]).strip(), "native_id": nid,
            "meta": {"answer_value": value, "aliases": aliases,
                     "normalized_value": a.get("normalized_value"), "answer_type": a.get("type")}}


def halluhard_row_to_record(r: dict[str, Any], domain: str, i: int, id_prefix: str) -> dict[str, Any]:
    """Map all four HalluHard domains without inventing unavailable references."""
    if domain == "research_questions":
        question, reference = r["research_question"], r["abstract"]
        native = str(r.get("doi") or r.get("arxiv_id") or i)
        extra = {"title": r.get("title"), "publication_year": r.get("publication_year"),
                 "judge_variant": "reference"}
    elif domain == "medical_guidelines":
        question, reference = r["question"], r["guideline_text"]
        native = str(r.get("source") or i)
        extra = {"source": r.get("source"), "judge_variant": "reference"}
    elif domain == "legal_cases":
        question, reference = r["question"], None
        native = str(r.get("case_id") or i)
        extra = {"source": r.get("source"), "question_category": r.get("question_category"),
                 "judge_variant": "knowledge"}
    elif domain == "coding":
        question, reference = r["prompt"], None
        native = str(r.get("task") or i)
        extra = {"language": r.get("language"), "task": r.get("task"),
                 "prompt_template": r.get("prompt_template"), "judge_variant": "coding"}
    else:
        raise ValueError(domain)
    return {"question_id": f"{id_prefix}_{domain[:2]}_{i:04d}", "question": str(question).strip(), "native_id": native,
            "meta": {"answer_value": None, "aliases": [], "reference": reference, "domain": domain, **extra}}


def _norm_q(q: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", q.lower()).split())


def dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out = []
    for r in rows:
        k = _norm_q(r["question"])
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


# ----------------------------------------------------------------------------- sampling
def stratified_sample(rows: list[dict[str, Any]], n_total: int, n_bins: int, seed: int, key: str = "s_pop") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drop duplicate question texts, assign `key` deciles over the *full* deduplicated set (rank-based, tie-safe),
    then sample n_total/n_bins questions per decile with RandomState(seed). Returns records with `s_pop_decile`."""
    import pandas as pd

    uniq = dedupe(rows)
    vals = np.array([float(r["meta"][key]) for r in uniq])
    ranks = pd.Series(vals).rank(method="first")
    dec = pd.qcut(ranks, n_bins, labels=False).to_numpy().astype(int)
    per = n_total // n_bins
    rng = np.random.RandomState(seed)
    chosen = []
    edges = {}
    for b in range(n_bins):
        idx = np.flatnonzero(dec == b)
        edges[str(b)] = {"n_pool": int(len(idx)), "s_pop_min": float(vals[idx].min()), "s_pop_max": float(vals[idx].max())}
        pick = rng.choice(idx, size=min(per, len(idx)), replace=False)
        for i in sorted(pick):
            rec = dict(uniq[i])
            rec["s_pop_decile"] = int(b)
            chosen.append(rec)
    info = {"n_source_rows": len(rows), "n_after_text_dedupe": len(uniq), "n_bins": n_bins, "per_bin": per, "seed": seed,
            "n_sampled": len(chosen), "bins": edges}
    return chosen, info


def random_sample(rows: list[dict[str, Any]], n_total: int | None, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Dedupe by question text, then take a plain seeded sample (or everything when n_total is None)."""
    uniq = dedupe(rows)
    info = {"n_source_rows": len(rows), "n_after_text_dedupe": len(uniq), "seed": seed}
    if n_total is None or n_total >= len(uniq):
        info.update({"n_sampled": len(uniq), "mode": "all"})
        return uniq, info
    rng = np.random.RandomState(seed)
    pick = sorted(rng.choice(len(uniq), size=n_total, replace=False))
    info.update({"n_sampled": len(pick), "mode": "random"})
    return [uniq[i] for i in pick], info


# ----------------------------------------------------------------------------- loaders
def load_popqa(cfg: Config, d: dict[str, Any]):
    from datasets import load_dataset

    ds = load_dataset(d["source"]["name"], split=d["source"]["split"])
    rows = [popqa_row_to_record(r, d["id_prefix"]) for r in ds]
    s = d["sampling"]
    return stratified_sample(rows, int(d["sampled_questions"]), int(s["bins"]), int(s["seed"]), s["stratify_by"])


def load_trivia_qa(cfg: Config, d: dict[str, Any]):
    from datasets import load_dataset

    src = d["source"]
    ds = load_dataset(src["name"], src.get("config"), split=src["split"])
    rows = [trivia_row_to_record(r, d["id_prefix"]) for r in ds]
    n = d.get("sampled_questions")
    return random_sample(rows, None if n is None else int(n), int(d["sampling"]["seed"]))


def load_hallu_hard(cfg: Config, d: dict[str, Any]):
    """Pool configured HalluHard domains, with optional deterministic caps per domain."""
    rows: list[dict[str, Any]] = []
    per_domain: dict[str, int] = {}
    per_domain_selected: dict[str, int] = {}
    caps = d.get("per_domain_questions") or {}
    seed = int(d["sampling"]["seed"])
    for domain, path in sorted(d["source"].items()):
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"HalluHard file missing: {p}")
        raw = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        recs = [halluhard_row_to_record(r, domain, i, d["id_prefix"]) for i, r in enumerate(raw)]
        per_domain[domain] = len(recs)
        cap = caps.get(domain)
        selected, _ = random_sample(recs, None if cap is None else int(cap), seed)
        per_domain_selected[domain] = len(selected)
        rows.extend(selected)
    n = d.get("sampled_questions")
    chosen, info = random_sample(rows, None if n is None else int(n), seed)
    info["per_domain_source_rows"] = per_domain
    info["per_domain_sampled"] = {k: sum(1 for r in chosen if r["meta"]["domain"] == k) for k in per_domain}
    info["per_domain_requested"] = {k: caps.get(k) for k in per_domain}
    info["per_domain_after_cap"] = per_domain_selected
    return chosen, info


LOADERS = {"popqa": load_popqa, "trivia_qa": load_trivia_qa, "hallu_hard": load_hallu_hard}


def load_questions(cfg: Config, dataset: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    d = cfg.dataset(dataset)
    try:
        loader = LOADERS[d["loader"]]
    except KeyError:
        raise SystemExit(f"unknown loader {d['loader']!r}; known: {sorted(LOADERS)}")
    qs, info = loader(cfg, d)
    info["dataset"] = cfg.dataset_name(dataset)
    info["loader"] = d["loader"]
    return qs, info


# ----------------------------------------------------------------------------- split
SPLIT_NAMES = ("train", "validation", "test")


def _split_fresh(questions: list[dict[str, Any]], cfg: Config, dataset: str | None = None) -> dict[str, list[str]]:
    """60/20/20 by question id; stratified within `sampling.stratify_by` when that key names a meta field
    (HalluHard: domain) so both domains keep their proportions in every split."""
    from .vendor.splits import make_question_split

    s = cfg["split"]
    fracs = (float(s["train"]), float(s["validation"]), float(s["test"]))
    seed = int(s["seed"])
    # NB: `sampling.stratify_by` selects the *sample* (PopQA: s_pop deciles) and must not be reused here —
    # grouping the split by a popularity integer makes thousands of singleton groups, each of which rounds
    # into train, skewing the split badly. Only an explicit categorical key stratifies the split.
    strat = cfg.dataset(dataset).get("split_stratify_by")
    if not strat or not all(strat in q["meta"] for q in questions):
        return make_question_split([q["question_id"] for q in questions], seed, fracs)
    groups: dict[str, list[str]] = {}
    for q in questions:
        groups.setdefault(str(q["meta"][strat]), []).append(q["question_id"])
    out: dict[str, list[str]] = {k: [] for k in SPLIT_NAMES}
    for k, ids in sorted(groups.items()):
        part = make_question_split(ids, seed, fracs)
        for name in out:
            out[name].extend(part[name])
    return {k: sorted(v) for k, v in out.items()}


def build_split(questions: list[dict[str, Any]], cfg: Config, dataset: str | None = None,
                existing: dict[str, Any] | None = None) -> dict[str, list[str]]:
    """Split assignment, additively.

    Enlarging a dataset's question sample must not move questions between splits: responses, probe
    fits and every published cell are tied to the split a question was in. So questions already
    assigned keep their assignment, and only genuinely new questions are split (with the same seed
    and stratification). A fresh dataset just takes the plain split.
    """
    if not existing:
        return _split_fresh(questions, cfg, dataset)
    prior: dict[str, str] = {qid: name for name in SPLIT_NAMES for qid in (existing.get(name) or [])}
    present = {q["question_id"] for q in questions}
    out: dict[str, list[str]] = {k: [] for k in SPLIT_NAMES}
    for qid, name in prior.items():
        if qid in present:
            out[name].append(qid)
    new = [q for q in questions if q["question_id"] not in prior]
    if new:
        part = _split_fresh(new, cfg, dataset)
        for name in out:
            out[name].extend(part[name])
    return {k: sorted(v) for k, v in out.items()}

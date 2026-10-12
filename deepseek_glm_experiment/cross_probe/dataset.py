"""Join replay index + generation statistics + verdicts + questions + split into the probe dataset (rows.parquet),
plus layer-major activation files so per-layer analyses are cheap and re-runnable."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .labels import DOMAIN_OF, state_for
from .prompts import DEFAULT_ERROR_FRAMING
from .vendor.io_utils import read_json, read_jsonl, sha256_text, write_json
from .vendor.splits import assert_no_leakage, split_lookup

FEATURE_COLS = ["n_answer_tokens", "mean_logprob", "min_logprob", "mean_entropy", "max_entropy"]
ROW_COLS = ["question_id", "conversation_id", "split", "condition", "template_id", "framing", "held_out_template", "state",
            "label", "domain", "row", "s_pop", "s_pop_decile", "o_pop", "prop", "question_domain", "response_text", "verdict",
            "verdict_source", "truncated"] + FEATURE_COLS


def replay_exclusion(value: Any) -> str | None:
    """A blank replay reason remains blank after pandas converts None to NaN."""
    return None if value is None or pd.isna(value) or value == "" else str(value)


def backfill_framing(df: pd.DataFrame) -> pd.Series:
    """`framing` names the variant a row was generated under: "none" for organic, "v1"/"strict" for the two
    truthful frames, and the error register otherwise. Rows produced before registers existed carry no column;
    those are all the phase-1 variants (`lie`, and the v1-framed truthful)."""
    f = df["framing"] if "framing" in df.columns else pd.Series("none", index=df.index)
    f = f.fillna("none")
    f = f.where(~((df["condition"] == "incorrect") & (f == "none")), DEFAULT_ERROR_FRAMING)
    return f.where(~((df["condition"] == "truthful") & (f == "none")), "v1")


def build_rows(run_dir: Path, questions_path: Path, split_file: Path,
               known_questions: set[str] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    index = pd.DataFrame(read_jsonl(run_dir / "replay" / "index.jsonl"))
    stats = pd.DataFrame(read_jsonl(run_dir / "replay" / "genstats.jsonl")).drop(
        columns=["question_id", "sample_id", "row", "truncated", "n_generated_tokens", "n_answer_tokens"], errors="ignore")
    verd = pd.DataFrame(read_jsonl(run_dir / "judge" / "verdicts.jsonl")).drop(columns=["question_id", "condition", "template_id"], errors="ignore")
    verd = verd.drop_duplicates("conversation_id", keep="last")
    gens = {g["conversation_id"]: g for g in read_jsonl(run_dir / "generations.jsonl")}
    qs = {q["question_id"]: q for q in read_jsonl(questions_path)}
    split = read_json(split_file)
    split = {k: split[k] for k in ("train", "validation", "test")}
    lookup = split_lookup(split)

    df = index.merge(stats, on="conversation_id", how="left").merge(verd, on="conversation_id", how="left", suffixes=("", "_v"))
    df["framing"] = backfill_framing(df)
    df["split"] = df["question_id"].map(lookup)
    df["held_out_template"] = [bool(gens[c].get("held_out_template")) for c in df["conversation_id"]]
    df["response_text"] = [gens[c]["response_text"] for c in df["conversation_id"]]
    # dataset-specific question metadata; absent for datasets without popularity (TriviaQA, AA, HalluHard)
    df["s_pop"] = [qs[q]["meta"].get("s_pop") for q in df["question_id"]]
    df["o_pop"] = [qs[q]["meta"].get("o_pop") for q in df["question_id"]]
    df["prop"] = [qs[q]["meta"].get("prop") for q in df["question_id"]]
    df["question_domain"] = [qs[q]["meta"].get("domain") for q in df["question_id"]]
    df["s_pop_decile"] = [qs[q].get("s_pop_decile") for q in df["question_id"]]
    df["pair_sha256"] = [sha256_text(gens[c]["prompt_text"] + "\n<<RESPONSE>>\n" + gens[c]["response_text"]) for c in df["conversation_id"]]

    funnel: dict[str, Any] = {"total_responses": int(len(df))}
    reasons: Counter = Counter()
    states, labels, domains, keep = [], [], [], np.ones(len(df), dtype=bool)
    for i, r in enumerate(df.itertuples(index=False)):
        d = r._asdict()
        why = None
        exclude_reason = replay_exclusion(d.get("exclude_reason"))
        if exclude_reason:
            why = f"replay:{exclude_reason}"
        elif not d["alignment_ok"]:
            why = "alignment_failure"
        elif bool(d["is_duplicate"]):
            why = "duplicate_response_within_question_condition_template"
        elif pd.isna(d.get("verdict")):
            why = "not_judged"
        elif pd.isna(d["split"]):
            why = "no_split"
        elif known_questions is not None and d["condition"] != "organic" and d["question_id"] not in known_questions:
            # instructed rows only count for questions the model demonstrably knows
            why = "question_not_in_known_set"
        st, lab, dom = None, None, None
        if why is None:
            st, lab, ex = state_for(d["condition"], d["verdict"], d["framing"])
            if ex:
                why = f"label:{ex}"
            else:
                dom = DOMAIN_OF[st]
        states.append(st); labels.append(lab); domains.append(dom)
        if why:
            reasons[why] += 1
            keep[i] = False
    df["state"], df["label"], df["domain"] = states, labels, domains
    df = df.rename(columns={"source": "verdict_source"})
    funnel["excluded"] = dict(reasons)
    kept = df[keep].copy()
    kept["label"] = kept["label"].astype(int)
    for c in FEATURE_COLS:
        kept[c] = kept[c].astype(float)
    funnel["usable_responses"] = int(len(kept))
    funnel["per_split_state"] = {s: {st: int(((kept["split"] == s) & (kept["state"] == st)).sum()) for st in ("O0", "O1", "I0", "I1")}
                                 for s in ("train", "validation", "test")}
    funnel["per_split_state_by_framing"] = {
        f: {s: {st: int(((kept["split"] == s) & (kept["state"] == st) & (kept["framing"] == f)).sum()) for st in ("O0", "O1", "I0", "I1")}
            for s in ("train", "validation", "test")}
        for f in sorted(set(kept["framing"])) if f != "none"}
    funnel["per_split_questions"] = {s: int(kept[kept["split"] == s]["question_id"].nunique()) for s in ("train", "validation", "test")}
    if len(kept):
        assert_no_leakage(kept[["question_id", "split", "pair_sha256"]].rename(columns={"pair_sha256": "response_sha256"}).to_dict("records"), split)
    kept = kept.sort_values("row").reset_index(drop=True)
    return kept, funnel


def write_layer_major(src_npy: Path, rows: np.ndarray, dst_npy: Path, log=None) -> tuple[int, int]:
    """acts[N_all, L, H] (row-major memmap) -> dst[L, len(rows), H] float32, reading each source row once, sequentially."""
    src = np.load(src_npy, mmap_mode="r")
    n_layers, hidden = src.shape[1], src.shape[2]
    dst = np.lib.format.open_memmap(dst_npy, mode="w+", dtype=np.float32, shape=(n_layers, len(rows), hidden))
    order = np.argsort(rows)
    for k, j in enumerate(order):
        dst[:, j, :] = src[int(rows[j])]
        if log and (k + 1) % 2000 == 0:
            log.info("layer-major %d/%d", k + 1, len(rows))
    dst.flush()
    del dst
    return n_layers, hidden

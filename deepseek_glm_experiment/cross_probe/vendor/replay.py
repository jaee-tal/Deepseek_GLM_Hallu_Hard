"""Teacher-forced replay of stored token sequences.

Produces, per response:
  * mean-pooled hidden states over the answer span for every layer  -> activations.npy [N, L+1, H]
  * generation statistics from the model's own next-token distribution (T=1, full vocab)
  * token-alignment check results
Also provides the activation-determinism check.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .families import Family
from .io_utils import PROJECT_ROOT, append_jsonl, normalize_response, read_jsonl, sha256_text, write_json, write_jsonl

log = logging.getLogger(__name__)


def alignment_check(rec: dict[str, Any], tokenizer, family: Family) -> tuple[bool, list[str]]:
    problems: list[str] = []
    p, g = rec["prompt_token_ids"], rec["generated_token_ids"]
    s, e = rec["answer_start"], rec["answer_end"]
    if tokenizer(rec["prompt_text"], add_special_tokens=False)["input_ids"] != p:
        problems.append("prompt_retokenize_mismatch")
    if tokenizer.decode(p, skip_special_tokens=False) != rec["prompt_text"]:
        problems.append("prompt_decode_mismatch")
    if not (0 <= s <= e <= len(g)):
        problems.append("span_out_of_bounds")
    else:
        if tokenizer.decode(g[s:e], skip_special_tokens=True) != rec["response_text"]:
            problems.append("response_decode_mismatch")
        if any(t in family.reasoning_marker_ids for t in g[s:e]):
            problems.append("reasoning_marker_in_span")
    if rec["response_token_start"] != len(p) + s or rec["response_token_end"] != len(p) + e:
        problems.append("absolute_offsets_mismatch")
    if any(t in set(family.eos_ids) for t in g):
        problems.append("eos_inside_generated")
    return (not problems), problems


def _inner_model(model):
    # All retained checkpoints expose the text stack as ``model``. Multimodal
    # wrappers may add one more language_model/text_model wrapper.
    inner = model.model
    if hasattr(inner, "layers"):
        return inner
    for name in ("language_model", "text_model", "model"):
        candidate = getattr(inner, name, None)
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate
    raise AttributeError(f"cannot find text transformer layers below {type(model).__name__}")


def model_input_device(model) -> torch.device:
    """Input device for single-device and Accelerate-sharded models."""
    try:
        return model.get_input_embeddings().weight.device
    except Exception:  # noqa: BLE001
        return model.device


@torch.inference_mode()
def replay_one(model, ids: list[int], s: int, e: int, logit_chunk: int = 128):
    """Return answer-pooled activations, generation statistics, logprobs, and entropies."""
    x = torch.tensor(ids, dtype=torch.long, device=model_input_device(model))[None]
    out = _inner_model(model)(input_ids=x, attention_mask=torch.ones_like(x), output_hidden_states=True, use_cache=False, return_dict=True)
    hs = out.hidden_states  # tuple(L+1) of [1, S, H]
    pooled = torch.stack([h[0, s:e].float().mean(0) for h in hs]).cpu().numpy()
    last = out.last_hidden_state[0]  # [S, H]
    # answer token at absolute position t is predicted from position t-1
    pred_pos = last[s - 1:e - 1]
    targets = x[0, s:e]
    lps, ents, top1 = [], [], []
    for k in range(0, pred_pos.shape[0], logit_chunk):
        logits = model.lm_head(pred_pos[k:k + logit_chunk]).float()
        logp = torch.log_softmax(logits, dim=-1)
        tgt = targets[k:k + logit_chunk]
        lps.append(logp.gather(1, tgt[:, None])[:, 0])
        ents.append(-(logp.exp() * logp).sum(-1))
        top1.append(logp.max(-1).values.exp())
    lp = torch.cat(lps).cpu().numpy() if lps else np.zeros(0, dtype=np.float32)
    en = torch.cat(ents).cpu().numpy() if ents else np.zeros(0, dtype=np.float32)
    t1 = torch.cat(top1).cpu().numpy() if top1 else np.zeros(0, dtype=np.float32)
    stats = {
        "n_answer_tokens": int(e - s),
        "sum_logprob": float(lp.sum()) if lp.size else float("nan"),
        "mean_logprob": float(lp.mean()) if lp.size else float("nan"),
        "min_logprob": float(lp.min()) if lp.size else float("nan"),
        "mean_entropy": float(en.mean()) if en.size else float("nan"),
        "max_entropy": float(en.max()) if en.size else float("nan"),
        "first_token_entropy": float(en[0]) if en.size else float("nan"),
        "mean_top1_prob": float(t1.mean()) if t1.size else float("nan"),
    }
    return pooled, stats, lp, en


def check_last_hidden_is_normed(model, ids: list[int]) -> dict[str, Any]:
    """Is hidden_states[-1] the raw output of the last block, or post final-norm?"""
    inner = _inner_model(model)
    captured = {}

    def hook(mod, inp, outp):
        captured["raw"] = (outp[0] if isinstance(outp, tuple) else outp).detach()

    h = inner.layers[-1].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            x = torch.tensor(ids, dtype=torch.long, device=model_input_device(model))[None]
            out = inner(input_ids=x, output_hidden_states=True, use_cache=False, return_dict=True)
            raw = captured["raw"].float()
            last = out.hidden_states[-1].float()
            normed = inner.norm(captured["raw"]).float()
            d_raw = (last - raw).abs().max().item()
            d_norm = (last - normed).abs().max().item()
    finally:
        h.remove()
    return {"max_abs_diff_vs_raw": d_raw, "max_abs_diff_vs_normed": d_norm, "last_hidden_state_is_normed": d_norm < d_raw}


@torch.inference_mode()
def determinism_check(model, records: list[dict[str, Any]], n_rows: int, seed: int, warn_abs: float, fail_abs: float, subprocess_cmd: list[str] | None = None) -> dict[str, Any]:
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(records), size=min(n_rows, len(records)), replace=False)
    per_layer_max = None
    rel_max = 0.0
    bit_identical = True
    for i in idx:
        r = records[i]
        ids = r["prompt_token_ids"] + r["generated_token_ids"]
        s, e = r["response_token_start"], r["response_token_end"]
        if e <= s:
            continue
        a, _, _, _ = replay_one(model, ids, s, e)
        b, _, _, _ = replay_one(model, ids, s, e)
        d = np.abs(a - b).max(axis=1)
        per_layer_max = d if per_layer_max is None else np.maximum(per_layer_max, d)
        rel_max = max(rel_max, float((np.abs(a - b).max(axis=1) / (np.linalg.norm(a, axis=1) + 1e-9)).max()))
        bit_identical &= bool(np.array_equal(a, b))
    res = {
        "n_rows": int(len(idx)),
        "in_process_per_layer_max_abs_diff": per_layer_max.tolist() if per_layer_max is not None else None,
        "in_process_max_abs_diff": float(per_layer_max.max()) if per_layer_max is not None else None,
        "in_process_max_rel_diff": rel_max,
        "bit_identical": bit_identical,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "status": "ok",
    }
    if per_layer_max is not None:
        m = float(per_layer_max.max())
        if m > fail_abs:
            res["status"] = "fail"
        elif m > warn_abs:
            res["status"] = "warn"
    return res


def grow_npy(path: Path, n_total: int) -> None:
    """Extend a C-contiguous .npy along axis 0 in place, rewriting only the shape field of its header.

    The header is zero-padded to a 64-byte boundary, so a new row count of the same textual width fits
    without moving any data — a 39 GB array grows instantly instead of being copied. Raises if the width
    would change, so the caller can fall back to a full re-extract."""
    import io
    from numpy.lib import format as npf

    with open(path, "r+b") as f:
        version = npf.read_magic(f)
        shape, fortran, dtype = npf.read_array_header_1_0(f) if version == (1, 0) else npf.read_array_header_2_0(f)
        data_off = f.tell()
        if fortran:
            raise ValueError(f"{path.name}: fortran-ordered arrays cannot be grown in place")
        if n_total < shape[0]:
            raise ValueError(f"{path.name}: refusing to shrink {shape[0]} -> {n_total}")
        new_shape = (n_total,) + tuple(shape[1:])
        buf = io.BytesIO()
        npf.write_array_header_1_0(buf, {"descr": npf.dtype_to_descr(dtype), "fortran_order": False, "shape": new_shape})
        if buf.tell() != data_off:
            raise ValueError(f"{path.name}: header width would change ({data_off} -> {buf.tell()}); re-extract instead")
        f.seek(0)
        f.write(buf.getvalue())
    os.truncate(path, data_off + int(np.prod(new_shape)) * dtype.itemsize)


def index_prefix_len(out_dir: Path, records: list[dict[str, Any]]) -> int:
    """How many leading records the existing replay index already covers, or -1 if it is not a prefix.

    Appending is only safe when the rows already on disk correspond, in order, to the first N generations —
    every `row` index and therefore every downstream `ds_row` then keeps its meaning."""
    idx_path = out_dir / "index.jsonl"
    if not idx_path.exists():
        return 0
    idx = read_jsonl(idx_path)
    if len(idx) > len(records):
        return -1
    for a, b in zip(idx, records):
        if a["conversation_id"] != b["conversation_id"]:
            return -1
    return len(idx)


def extract_all(
    model,
    tokenizer,
    family: Family,
    records: list[dict[str, Any]],
    out_dir: Path,
    logit_chunk: int,
    allow_alignment_failures: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(records)
    start = index_prefix_len(out_dir, records) if resume else 0
    if start < 0:
        raise SystemExit("replay index is not a prefix of generations.jsonl — rows were reordered or removed; "
                         "re-run 03 with --force to re-extract everything")
    # probe shapes from a first forward pass
    first = next(r for r in records if r["answer_end"] > r["answer_start"])
    ids0 = first["prompt_token_ids"] + first["generated_token_ids"]
    pooled0, _, _, _ = replay_one(model, ids0, first["response_token_start"], first["response_token_end"], logit_chunk)
    n_layers, hidden = pooled0.shape
    act_path = out_dir / "activations.npy"
    if start:
        log.info("resuming replay: %d rows already extracted, %d new", start, n - start)
        grow_npy(act_path, n)
        acts = np.load(act_path, mmap_mode="r+")
        if acts.shape[1:] != (n_layers, hidden):
            raise SystemExit(f"stored activations are {acts.shape[1:]}, this model gives {(n_layers, hidden)}")
    else:
        acts = np.lib.format.open_memmap(act_path, mode="w+", dtype=np.float32, shape=(n, n_layers, hidden))
    index_rows, stats_rows = [], []
    tok_lp: dict[str, np.ndarray] = {}
    tok_en: dict[str, np.ndarray] = {}
    n_align_fail = 0
    seen_hash_by_q: dict[str, dict[str, str]] = {}
    seen_hash_global: dict[str, str] = {}
    if start:
        # Duplicate detection must span the boundary: a new row repeating an earlier response is still a duplicate.
        index_rows = read_jsonl(out_dir / "index.jsonl")
        stats_rows = read_jsonl(out_dir / "genstats.jsonl")
        for prev in index_rows:
            k = f"{prev['question_id']}|{prev.get('condition')}|{prev.get('framing', 'none')}|{prev.get('template_id')}"
            seen_hash_by_q.setdefault(k, {}).setdefault(prev["normalized_sha256"], prev["conversation_id"])
            seen_hash_global.setdefault(prev["normalized_sha256"], prev["conversation_id"])
        n_align_fail = sum(1 for prev in index_rows if not prev["alignment_ok"])
    for i, r in enumerate(records[start:], start=start):
        ok, problems = alignment_check(r, tokenizer, family)
        exclude = r.get("exclude_reason")
        if not ok:
            n_align_fail += 1
            exclude = exclude or "alignment_failure"
        norm_hash = sha256_text(normalize_response(r["response_text"]))
        q = r["question_id"]
        qkey = f"{q}|{r.get('condition')}|{r.get('framing', 'none')}|{r.get('template_id')}"
        dup_within = norm_hash in seen_hash_by_q.setdefault(qkey, {})
        dup_global_of = seen_hash_global.get(norm_hash)
        cross_q_dup = (dup_global_of is not None) and not dup_within and seen_hash_global.get(norm_hash, "").split("__s")[0] != q
        is_dup = dup_within          # identical prompt + identical text => identical activations; cross-question collisions are kept
        seen_hash_by_q[qkey].setdefault(norm_hash, r["conversation_id"])
        seen_hash_global.setdefault(norm_hash, r["conversation_id"])
        s, e = r["response_token_start"], r["response_token_end"]
        if exclude is None and e > s:
            ids = r["prompt_token_ids"] + r["generated_token_ids"]
            pooled, stats, lp, en = replay_one(model, ids, s, e, logit_chunk)
            acts[i] = pooled
            tok_lp[r["conversation_id"]] = lp.astype(np.float32)
            tok_en[r["conversation_id"]] = en.astype(np.float32)
        else:
            acts[i] = np.nan
            stats = {"n_answer_tokens": int(max(0, e - s))}
        if e <= s and exclude is None:
            exclude = "empty_answer"
        index_rows.append({
            "row": i,
            "model": r["model"],
            "domain": r["domain"],
            "question_id": q,
            "sample_id": r["sample_id"],
            "conversation_id": r["conversation_id"],
            "condition": r.get("condition"),
            "template_id": r.get("template_id"),
            "framing": r.get("framing", "none"),
            "response_sha256": r["response_sha256"],
            "pair_sha256": sha256_text(r["prompt_text"] + "\n<<RESPONSE>>\n" + r["response_text"]),
            "normalized_sha256": norm_hash,
            "cross_question_duplicate": bool(cross_q_dup),
            "n_answer_tokens": int(max(0, e - s)),
            "n_generated_tokens": r["n_generated"],
            "truncated": r["truncated"],
            "alignment_ok": ok,
            "alignment_problems": problems,
            "is_duplicate": is_dup,
            "duplicate_of": seen_hash_by_q[qkey][norm_hash] if is_dup else None,
            "exclude_reason": exclude,
        })
        stats_rows.append({"row": i, "conversation_id": r["conversation_id"], "question_id": q, "sample_id": r["sample_id"],
                           "n_generated_tokens": r["n_generated"], "response_chars": len(r["response_text"]),
                           "truncated": r["truncated"], **stats})
        if (i + 1) % 50 == 0:
            log.info("replay %d/%d", i + 1, n)
            acts.flush()
    acts.flush()
    del acts
    write_jsonl(out_dir / "index.jsonl", index_rows)
    write_jsonl(out_dir / "genstats.jsonl", stats_rows)
    tok_path = out_dir / ("token_stats.npz" if not start else f"token_stats_from_{start:06d}.npz")
    np.savez_compressed(tok_path, **{f"lp__{k}": v for k, v in tok_lp.items()}, **{f"en__{k}": v for k, v in tok_en.items()})
    n_dup = sum(1 for r in index_rows if r["is_duplicate"])
    n_cross = sum(1 for r in index_rows if r.get("cross_question_duplicate"))
    summary = {
        "n_records": n,
        "n_layers_incl_embeddings": int(n_layers),
        "hidden_size": int(hidden),
        "layer_index_convention": "0 = embedding output; l = output of decoder block l",
        "n_alignment_failures": n_align_fail,
        "n_duplicates_within_question": n_dup,
        "n_cross_question_text_collisions_kept": n_cross,
        "n_excluded": sum(1 for r in index_rows if r["exclude_reason"]),
        "exclude_reasons": _count([r["exclude_reason"] for r in index_rows if r["exclude_reason"]]),
    }
    if n_align_fail and not allow_alignment_failures:
        summary["status"] = "alignment_failures"
    else:
        summary["status"] = "ok"
    return summary


def _count(xs):
    from collections import Counter

    return dict(Counter(xs))

"""Immutable-shard teacher-forced replay with deterministic resume/materialization."""
from __future__ import annotations

import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..vendor.families import Family
from ..vendor.io_utils import (normalize_response, read_json, read_jsonl, sha256_file,
                               sha256_text, write_json, write_jsonl)
from ..vendor.replay import alignment_check, replay_one

log = logging.getLogger(__name__)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _valid_shards(chunk_root: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifests = []
    cursor = 0
    for path in sorted(chunk_root.glob("*/manifest.json")) if chunk_root.exists() else []:
        item = read_json(path)
        start, end = int(item["start"]), int(item["end"])
        if start != cursor or end > len(records):
            raise RuntimeError(f"non-contiguous replay shard {path}: expected start {cursor}")
        expected = [r["conversation_id"] for r in records[start:end]]
        if item["conversation_ids"] != expected:
            raise RuntimeError(f"replay shard {path} does not match current generations")
        for name, digest in item["sha256"].items():
            if sha256_file(path.parent / name) != digest:
                raise RuntimeError(f"replay shard checksum failed: {path.parent / name}")
        manifests.append(item)
        cursor = end
    return manifests


def _prior_duplicate_state(chunk_root: Path, manifests: list[dict[str, Any]]):
    by_prompt: dict[str, dict[str, str]] = {}
    global_seen: dict[str, str] = {}
    for manifest in manifests:
        folder = chunk_root / manifest["folder"]
        for row in read_jsonl(folder / "index.jsonl"):
            key = f"{row['question_id']}|{row.get('condition')}|{row.get('framing', 'none')}|{row.get('template_id')}"
            by_prompt.setdefault(key, {}).setdefault(row["normalized_sha256"], row["conversation_id"])
            global_seen.setdefault(row["normalized_sha256"], row["conversation_id"])
    return by_prompt, global_seen


def extract_shards(model, tokenizer, family: Family, records: list[dict[str, Any]], out_dir: Path,
                   chunk_rows: int, logit_chunk: int, expected_layers: int | None = None,
                   expected_hidden: int | None = None,
                   progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_root = out_dir / "chunks"
    chunk_root.mkdir(exist_ok=True)
    prior = _valid_shards(chunk_root, records)
    start = prior[-1]["end"] if prior else 0
    by_prompt, global_seen = _prior_duplicate_state(chunk_root, prior)

    is_vllm = getattr(model, "cross_probe_backend", None) == "vllm"
    first = next(r for r in records if r["answer_end"] > r["answer_start"] and not r.get("exclude_reason"))
    if is_vllm:
        n_hidden_states, hidden = int(model.n_hidden_states), int(model.hidden_size)
    else:
        ids = first["prompt_token_ids"] + first["generated_token_ids"]
        probe, _, _, _ = replay_one(model, ids, first["response_token_start"], first["response_token_end"], logit_chunk)
        n_hidden_states, hidden = probe.shape
    if expected_layers is not None and n_hidden_states != expected_layers + 1:
        raise RuntimeError(f"expected embedding + {expected_layers} text layers, got {n_hidden_states} hidden states")
    if expected_hidden is not None and hidden != expected_hidden:
        raise RuntimeError(f"expected hidden size {expected_hidden}, got {hidden}")

    log.info("chunked replay: %d records, %d already complete, chunk size %d", len(records), start, chunk_rows)
    for lo in range(start, len(records), chunk_rows):
        hi = min(len(records), lo + chunk_rows)
        rows = records[lo:hi]
        acts = np.full((len(rows), n_hidden_states, hidden), np.nan, dtype=np.float32)
        index_rows, stat_rows = [], []
        token_payload: dict[str, np.ndarray] = {}
        batch_replays = {}
        if is_vllm:
            replayable = []
            for rec in rows:
                ok, _ = alignment_check(rec, tokenizer, family)
                if ok and not rec.get("exclude_reason") and int(rec["response_token_end"]) > int(rec["response_token_start"]):
                    replayable.append(rec)
            if replayable:
                batch_replays = model.replay_batch(replayable)
        for offset, rec in enumerate(rows):
            row_number = lo + offset
            ok, problems = alignment_check(rec, tokenizer, family)
            exclude = rec.get("exclude_reason") or (None if ok else "alignment_failure")
            normalized = sha256_text(normalize_response(rec["response_text"]))
            qid = rec["question_id"]
            prompt_key = f"{qid}|{rec.get('condition')}|{rec.get('framing', 'none')}|{rec.get('template_id')}"
            within = normalized in by_prompt.setdefault(prompt_key, {})
            global_prior = global_seen.get(normalized)
            cross = global_prior is not None and not within and not global_prior.startswith(f"{qid}__")
            duplicate_of = by_prompt[prompt_key].get(normalized)
            by_prompt[prompt_key].setdefault(normalized, rec["conversation_id"])
            global_seen.setdefault(normalized, rec["conversation_id"])
            s, e = int(rec["response_token_start"]), int(rec["response_token_end"])
            if e <= s and exclude is None:
                exclude = "empty_answer"
            if exclude is None:
                if is_vllm:
                    pooled, stats, lp, entropy = batch_replays[rec["conversation_id"]]
                else:
                    full_ids = rec["prompt_token_ids"] + rec["generated_token_ids"]
                    pooled, stats, lp, entropy = replay_one(model, full_ids, s, e, logit_chunk)
                acts[offset] = pooled
                token_payload[f"lp__{rec['conversation_id']}"] = lp.astype(np.float32)
                token_payload[f"en__{rec['conversation_id']}"] = entropy.astype(np.float32)
            else:
                stats = {"n_answer_tokens": max(0, e - s)}
            index_rows.append({
                "row": row_number, "model": rec["model"], "domain": rec["domain"], "question_id": qid,
                "sample_id": rec["sample_id"], "conversation_id": rec["conversation_id"],
                "condition": rec.get("condition"), "template_id": rec.get("template_id"),
                "framing": rec.get("framing", "none"), "response_sha256": rec["response_sha256"],
                "pair_sha256": sha256_text(rec["prompt_text"] + "\n<<RESPONSE>>\n" + rec["response_text"]),
                "normalized_sha256": normalized, "cross_question_duplicate": bool(cross),
                "n_answer_tokens": max(0, e - s), "n_generated_tokens": rec["n_generated"],
                "truncated": rec["truncated"], "alignment_ok": ok, "alignment_problems": problems,
                "is_duplicate": bool(within), "duplicate_of": duplicate_of, "exclude_reason": exclude,
            })
            stat_rows.append({"row": row_number, "conversation_id": rec["conversation_id"],
                              "question_id": qid, "sample_id": rec["sample_id"],
                              "n_generated_tokens": rec["n_generated"],
                              "response_chars": len(rec["response_text"]),
                              "truncated": rec["truncated"], **stats})
        folder_name = f"{lo:08d}_{hi:08d}"
        final = chunk_root / folder_name
        tmp = chunk_root / (folder_name + ".tmp")
        if tmp.exists():
            import shutil
            shutil.rmtree(tmp)
        tmp.mkdir()
        _atomic_npy(tmp / "activations.npy", acts)
        write_jsonl(tmp / "index.jsonl", index_rows)
        write_jsonl(tmp / "genstats.jsonl", stat_rows)
        with (tmp / "token_stats.npz.tmp").open("wb") as handle:
            np.savez_compressed(handle, **token_payload)
        os.replace(tmp / "token_stats.npz.tmp", tmp / "token_stats.npz")
        names = ("activations.npy", "index.jsonl", "genstats.jsonl", "token_stats.npz")
        manifest = {"version": 1, "folder": folder_name, "start": lo, "end": hi,
                    "conversation_ids": [r["conversation_id"] for r in rows],
                    "shape": list(acts.shape), "sha256": {name: sha256_file(tmp / name) for name in names}}
        write_json(tmp / "manifest.json", manifest)
        os.replace(tmp, final)
        if progress:
            progress(hi, len(records))
        log.info("replay shard %d:%d complete", lo, hi)

    manifests = _valid_shards(chunk_root, records)
    index = [row for m in manifests for row in read_jsonl(chunk_root / m["folder"] / "index.jsonl")]
    segment = (["embedding"] + ["encoder"] * min(20, expected_layers or 0)
               + ["decoder"] * max(0, (expected_layers or n_hidden_states - 1) - 20))
    if expected_layers != 40:
        segment = ["embedding"] + ["text_block"] * (n_hidden_states - 1)
    summary = {
        "status": "ok" if not any(not r["alignment_ok"] for r in index) else "alignment_failures",
        "n_records": len(records), "n_layers_incl_embeddings": n_hidden_states,
        "n_text_layers": n_hidden_states - 1, "hidden_size": hidden,
        "layer_index_convention": "0 = embedding output; l = output of text block l",
        "layer_segments": segment,
        "n_alignment_failures": sum(not r["alignment_ok"] for r in index),
        "n_duplicates_within_question": sum(r["is_duplicate"] for r in index),
        "n_cross_question_text_collisions_kept": sum(r["cross_question_duplicate"] for r in index),
        "n_excluded": sum(bool(r["exclude_reason"]) for r in index),
        "exclude_reasons": dict(Counter(r["exclude_reason"] for r in index if r["exclude_reason"])),
        "n_shards": len(manifests),
    }
    write_json(out_dir / "checks.json", {"extraction": summary})
    return summary


def materialize(out_dir: Path, records: list[dict[str, Any]], force: bool = False) -> dict[str, Any]:
    """Build legacy-compatible files from immutable shards; safe to regenerate."""
    chunk_root = out_dir / "chunks"
    manifests = _valid_shards(chunk_root, records)
    if not manifests or manifests[-1]["end"] != len(records):
        raise RuntimeError(f"replay shards cover {manifests[-1]['end'] if manifests else 0}/{len(records)} rows")
    shape = manifests[0]["shape"]
    target = out_dir / "activations.npy"
    if target.exists() and not force:
        existing = np.load(target, mmap_mode="r")
        if existing.shape == (len(records), shape[1], shape[2]):
            return {"n_records": len(records), "shape": list(existing.shape), "reused": True}
    tmp = out_dir / "activations.tmp.npy"
    dst = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32,
                                    shape=(len(records), shape[1], shape[2]))
    index_rows, stat_rows = [], []
    for item in manifests:
        folder = chunk_root / item["folder"]
        dst[item["start"]:item["end"]] = np.load(folder / "activations.npy", mmap_mode="r")
        index_rows.extend(read_jsonl(folder / "index.jsonl"))
        stat_rows.extend(read_jsonl(folder / "genstats.jsonl"))
    dst.flush()
    del dst
    os.replace(tmp, target)
    write_jsonl(out_dir / "index.jsonl", index_rows)
    write_jsonl(out_dir / "genstats.jsonl", stat_rows)
    return {"n_records": len(records), "shape": [len(records), shape[1], shape[2]], "reused": False}

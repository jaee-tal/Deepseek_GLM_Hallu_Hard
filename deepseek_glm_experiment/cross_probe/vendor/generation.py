"""Batched sampling from a local HF model; writes generations.jsonl (one record per response).

Seed scheme ("per_batch_v1"): one torch seed per generate() call, derived from the base seed and the
batch index. Batched sampling shares a single RNG stream across the batch, so seeds are provenance,
not a replay mechanism: the stored token ids are the ground truth and all later stages replay them
with teacher forcing.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import torch

from .families import Family
from .io_utils import append_jsonl, iter_jsonl, sha256_text
from .replay import model_input_device

log = logging.getLogger(__name__)


def make_generation_config(gen_cfg: dict[str, Any], family: Family):
    from transformers import GenerationConfig

    gc = GenerationConfig(
        do_sample=bool(gen_cfg["do_sample"]),
        temperature=float(gen_cfg["temperature"]),
        top_p=float(gen_cfg["top_p"]),
        top_k=int(gen_cfg.get("top_k", 0)),
        repetition_penalty=1.0,
        max_new_tokens=int(gen_cfg["max_new_tokens"]),
        eos_token_id=list(family.eos_ids),
        pad_token_id=int(family.pad_id),
        return_dict_in_generate=True,
        output_scores=False,
    )
    return gc


def generation_config_hash(gc) -> str:
    return hashlib.sha256(json.dumps(gc.to_dict(), sort_keys=True, default=str).encode()).hexdigest()[:16]


def batch_seed(base_seed: int, batch_index: int) -> int:
    return (base_seed * 1_000_003 + batch_index) % (2**31)


def render_prompt(tokenizer, family: Family, messages: list[dict[str, str]]) -> tuple[str, list[int]]:
    # Keep duck-typed test/custom families compatible with the original API.
    text = (family.render_messages(messages) if hasattr(family, "render_messages") else
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                          **family.template_kwargs))
    if not text.endswith(family.prompt_tail):
        raise RuntimeError(f"rendered prompt does not end with expected tail {family.prompt_tail!r}: ...{text[-120:]!r}")
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return text, list(ids)


def cut_at_eos(gen: list[int], eos_ids: list[int], pad_id: int) -> tuple[list[int], str]:
    stop = set(eos_ids) | {pad_id}
    for i, t in enumerate(gen):
        if t in stop:
            return gen[:i], "eos"
    return gen, "length"


def existing_conversation_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {r["conversation_id"] for r in iter_jsonl(path)}


def _sorted_groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group rows by question, sort groups by prompt length (desc) to reduce padding."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(f"{r['question_id']}|{r.get('condition')}|{r.get('framing')}|{r.get('template_id')}", []).append(r)
    ordered = sorted(groups.values(), key=lambda g: -len(g[0]["prompt_token_ids"]))
    return ordered


@torch.inference_mode()
def generate_all(
    model,
    tokenizer,
    family: Family,
    jobs: list[dict[str, Any]],
    gen_cfg: dict[str, Any],
    out_path: Path,
    model_name: str,
    domain: str,
    batch_size: int,
) -> None:
    """jobs: dicts with question_id, sample_id, conversation_id, messages, prompt_text, prompt_token_ids."""
    if getattr(model, "cross_probe_backend", None) == "vllm":
        model.generate_all(tokenizer, family, jobs, gen_cfg, out_path, model_name, domain, batch_size)
        return
    gc = make_generation_config(gen_cfg, family)
    gc_hash = generation_config_hash(gc)
    done = existing_conversation_ids(out_path)
    todo = [j for j in jobs if j["conversation_id"] not in done]
    log.info("generation: %d jobs, %d already done, %d to do", len(jobs), len(done), len(todo))
    if not todo:
        return

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = family.pad_id

    # Build batches from the complete deterministic plan, then skip completed
    # batches. Global batch indices and seeds therefore do not shift on resume.
    # The cloud models use batch_size=1, which also makes a checkpoint captured
    # during a JSONL append exactly reproducible at the next conversation.
    flat = [r for g in _sorted_groups(jobs) for r in g]
    batches = [flat[i:i + batch_size] for i in range(0, len(flat), batch_size)]
    base_seed = int(gen_cfg["base_seed"])
    n_done = 0
    work_batches = [(b_idx, [r for r in batch if r["conversation_id"] not in done])
                    for b_idx, batch in enumerate(batches)]
    work_batches = [(b_idx, batch) for b_idx, batch in work_batches if batch]
    for work_idx, (b_idx, batch) in enumerate(work_batches):
        t0 = time.time()
        cur_bs = len(batch)
        records = None
        while records is None:
            try:
                if cur_bs >= len(batch):
                    records = _run_batch(model, tokenizer, family, batch, gc, gc_hash, base_seed, b_idx, model_name, domain, gen_cfg)
                else:
                    recs = []
                    for k in range(0, len(batch), cur_bs):
                        sub = batch[k:k + cur_bs]
                        recs.extend(_run_batch(model, tokenizer, family, sub, gc, gc_hash, base_seed, b_idx, model_name, domain, gen_cfg, sub_index=1 + k // cur_bs))
                    records = recs
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if cur_bs == 1:
                    raise
                cur_bs = max(1, cur_bs // 2)
                log.warning("OOM at batch %d; retrying with sub-batches of %d", b_idx, cur_bs)
        append_jsonl(out_path, records)
        n_done += len(records)
        log.info("batch %d/%d (plan batch %d; %d rows) in %.1fs — %d/%d done",
                 work_idx + 1, len(work_batches), b_idx, len(records), time.time() - t0, n_done, len(todo))


def _run_batch(model, tokenizer, family, batch, gc, gc_hash, base_seed, b_idx, model_name, domain, gen_cfg, sub_index: int = 0):
    seed = batch_seed(base_seed, b_idx * 1000 + sub_index) if sub_index else batch_seed(base_seed, b_idx)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    texts = [r["prompt_text"] for r in batch]
    enc = tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt")
    input_device = model_input_device(model)
    input_ids = enc["input_ids"].to(input_device)
    attn = enc["attention_mask"].to(input_device)
    prompt_len_padded = input_ids.shape[1]
    # prompt alignment: padded row with pads stripped must equal stored prompt ids
    for i, r in enumerate(batch):
        row = input_ids[i][attn[i].bool()].tolist()
        if row != r["prompt_token_ids"]:
            raise RuntimeError(f"prompt alignment failure for {r['conversation_id']}")
    out = model.generate(input_ids=input_ids, attention_mask=attn, generation_config=gc)
    seqs = out.sequences
    records = []
    members = [[r["question_id"], r["sample_id"]] for r in batch]
    for i, r in enumerate(batch):
        gen_raw = seqs[i, prompt_len_padded:].tolist()
        gen, stop_reason = cut_at_eos(gen_raw, family.eos_ids, family.pad_id)
        span = family.answer_span(gen)
        n_prompt = len(r["prompt_token_ids"])
        response_text = tokenizer.decode(gen[span.start:span.end], skip_special_tokens=True)
        full_text = tokenizer.decode(gen, skip_special_tokens=False)
        records.append({
            "model": model_name,
            "domain": domain,
            "question_id": r["question_id"],
            "sample_id": r["sample_id"],
            "conversation_id": r["conversation_id"],
            "condition": r.get("condition"),
            "template_id": r.get("template_id"),
            "framing": r.get("framing", "none"),
            "instruction_type": r.get("instruction_type"),
            "held_out_template": r.get("held_out_template"),
            "split": r.get("split"),
            "seed": seed,
            "seed_scheme": "per_batch_v1",
            "base_seed": base_seed,
            "batch_index": b_idx,
            "sub_batch_index": sub_index,
            "position_in_batch": i,
            "batch_members": members,
            "prompt_text": r["prompt_text"],
            "prompt_token_ids": r["prompt_token_ids"],
            "generated_token_ids": gen,
            "n_generated": len(gen),
            "answer_start": span.start,
            "answer_end": span.end,
            "response_token_start": n_prompt + span.start,
            "response_token_end": n_prompt + span.end,
            "n_answer_tokens": span.end - span.start,
            "response_text": response_text,
            "full_generation_text": full_text,
            "stop_reason": stop_reason,
            "truncated": stop_reason == "length",
            "exclude_reason": span.exclude_reason,
            "span_flags": span.flags,
            "response_sha256": sha256_text(response_text),
            "temperature": float(gen_cfg["temperature"]),
            "top_p": float(gen_cfg["top_p"]),
            "top_k": int(gen_cfg.get("top_k", 0)),
            "max_new_tokens": int(gen_cfg["max_new_tokens"]),
            "generation_config_hash": gc_hash,
        })
    return records

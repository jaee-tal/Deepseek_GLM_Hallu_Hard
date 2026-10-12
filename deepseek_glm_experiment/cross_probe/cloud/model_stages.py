"""GPU stages that keep one very large subject model loaded across all datasets."""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from ..config import Config, questions_path, split_path
from ..jobs import build_jobs, error_samples_for
from ..vendor import provenance
from ..vendor.families import make_family
from ..vendor.generation import generate_all
from ..vendor.io_utils import read_json, read_jsonl, sha256_file, write_json
from ..vendor.modeling import (check_quantization, load_model_and_tokenizer, model_commit_sha,
                               setup_hf_env)
from ..vendor.replay import replay_one
from ..vendor.splits import split_lookup
from .replay import extract_shards, materialize
from .vllm_backend import VllmSubject

log = logging.getLogger(__name__)


def load_subject(cfg: Config, model_name: str, *, extract_hidden_states: bool = False):
    setup_hf_env(cfg.hf_home())
    mcfg = cfg.model(model_name)
    if mcfg.get("loader") == "vllm":
        model = VllmSubject(mcfg, extract_hidden_states=extract_hidden_states)
        tokenizer = model.tokenizer
        qinfo = model.quantization_info()
    else:
        model, tokenizer = load_model_and_tokenizer(mcfg)
        qinfo = check_quantization(model, mcfg)
    family = make_family(mcfg["family"], tokenizer, mcfg.get("template_kwargs", {}), mcfg)
    return model, tokenizer, family, mcfg, qinfo


def model_smoke(cfg: Config, model_name: str, root: Path, model, tokenizer, family, mcfg) -> dict:
    """Generation, final-span, replay-shape and determinism gate."""
    gen = dict(cfg.generation(model_name, "pop_qa"))
    gen.update({"organic_samples_per_question": 1, "max_new_tokens": min(512, int(gen["max_new_tokens"])),
                "batch_size": 1})
    question = [{"question_id": "preflight_q", "question": "What is the capital of France?"}]
    jobs = build_jobs(question, {"preflight_q": "train"}, gen, tokenizer, family,
                      conditions="organic", n_organic=1)
    output = root / "preflight_generation.jsonl"
    output.unlink(missing_ok=True)
    generate_all(model, tokenizer, family, jobs, gen, output, mcfg["checkpoint"], "preflight", 1)
    record = read_jsonl(output)[0]
    if record["exclude_reason"] or record["answer_end"] <= record["answer_start"]:
        raise RuntimeError(f"preflight did not produce a final answer span: {record['exclude_reason']}")
    expected = (int(mcfg["expected_text_layers"]) + 1, int(mcfg["expected_hidden_size"]))
    if getattr(model, "cross_probe_backend", None) == "vllm":
        if model.extract_hidden_states:
            a = model.replay_batch([record])[record["conversation_id"]][0]
            b = model.replay_batch([record])[record["conversation_id"]][0]
            if a.shape != expected:
                raise RuntimeError(f"preflight activation shape {a.shape}, expected {expected}")
            max_abs = float(np.max(np.abs(a - b)))
            per_layer_abs = np.max(np.abs(a - b), axis=1)
            max_rel = float(np.max(per_layer_abs / (np.linalg.norm(a, axis=1) + 1e-9)))
            rel_limit = float(mcfg.get("vllm_replay_determinism_fail_rel", 1.0e-2))
            if max_rel > rel_limit:
                raise RuntimeError(
                    f"preflight replay relative nondeterminism {max_rel:g} exceeds {rel_limit:g} "
                    f"(maximum absolute difference {max_abs:g})"
                )
            activation_shape = list(a.shape)
        else:
            # Generation stages use the lean vLLM engine.  The activation stage
            # and explicit preflight separately validate every residual layer.
            max_abs, max_rel, activation_shape = None, None, list(expected)
    else:
        ids = record["prompt_token_ids"] + record["generated_token_ids"]
        a, _, _, _ = replay_one(model, ids, record["response_token_start"], record["response_token_end"],
                                int(cfg["replay"]["logit_chunk"]))
        b, _, _, _ = replay_one(model, ids, record["response_token_start"], record["response_token_end"],
                                int(cfg["replay"]["logit_chunk"]))
        if a.shape != expected:
            raise RuntimeError(f"preflight activation shape {a.shape}, expected {expected}")
        max_abs = float(np.max(np.abs(a - b)))
        max_rel = float(np.max(np.max(np.abs(a - b), axis=1) /
                               (np.linalg.norm(a, axis=1) + 1e-9)))
        if max_abs > float(cfg["replay"]["determinism_fail_abs"]):
            raise RuntimeError(f"preflight replay nondeterminism {max_abs:g} exceeds failure threshold")
        activation_shape = list(a.shape)
    result = {"status": "ok", "activation_shape": activation_shape, "max_abs_replay_diff": max_abs,
              "max_relative_replay_diff": max_rel,
              "n_answer_tokens": record["n_answer_tokens"], "response_text": record["response_text"],
              "model_commit_sha": model_commit_sha(mcfg["checkpoint"], mcfg.get("revision"))}
    write_json(root / "model_preflight.json", result)
    return result


def _run_dir(cfg: Config, experiment_id: str, model_name: str, dataset: str) -> Path:
    path = cfg.run_dir(experiment_id, model_name, dataset)
    path.mkdir(parents=True, exist_ok=True)
    (path / "logs").mkdir(exist_ok=True)
    return path


def generate_stage(cfg: Config, experiment_id: str, model_name: str, conditions: str,
                   model, tokenizer, family, mcfg, qinfo,
                   progress: Callable[[dict], None] | None = None) -> None:
    cloud = cfg["cloud"]
    for index, dataset in enumerate(cloud["datasets"]):
        dcfg = cfg.dataset(dataset)
        run_dir = _run_dir(cfg, experiment_id, model_name, dataset)
        questions = read_jsonl(questions_path(cfg, dataset))
        split = read_json(split_path(cfg, dataset))
        lookup = split_lookup({name: split[name] for name in ("train", "validation", "test")})
        used = {"n": len(questions), "question_ids": [q["question_id"] for q in questions],
                "questions_sha256": sha256_file(questions_path(cfg, dataset)),
                "split_sha256": sha256_file(split_path(cfg, dataset))}
        used_path = run_dir / "questions_used.json"
        if used_path.exists() and read_json(used_path) != used:
            raise RuntimeError(f"question/split mismatch with prior {model_name}/{dataset} stage")
        write_json(used_path, used)
        known = None
        n_error = None
        if conditions == "inorganic":
            known_meta = read_json(run_dir / "known_questions.json")
            if known_meta.get("criterion") != "correct_never_wrong":
                raise RuntimeError(f"{dataset} eligibility is not correct_never_wrong")
            known = set(known_meta["question_ids"])
            n_error = error_samples_for(len(known), int(cfg.generation(model_name, dataset).get("error_rows_target", 8000)))
        gen_cfg = cfg.generation(model_name, dataset)
        jobs = build_jobs(questions, lookup, gen_cfg, tokenizer, family, style=dcfg["prompt_style"],
                          conditions=conditions, known_questions=known, n_error=n_error)
        generate_all(model, tokenizer, family, jobs, gen_cfg, run_dir / "generations.jsonl",
                     mcfg["checkpoint"], dataset, int(gen_cfg["batch_size"]))
        provenance.write(run_dir, provenance.collect(cfg.hash, str(cfg.path), mcfg, extra={
            "cloud_v3": True, "experiment_id": experiment_id, "model_name": model_name,
            "dataset": dataset, "conditions": conditions, "quantization_check": qinfo,
            "model_commit_sha": model_commit_sha(mcfg["checkpoint"], mcfg.get("revision")), "n_jobs": len(jobs),
            "n_known_questions": len(known) if known is not None else None,
            "error_samples_per_question": n_error,
        }), section=f"generation_{conditions}")
        state = {"stage": f"{conditions}_generation", "datasets_complete": index + 1,
                 "datasets_total": len(cloud["datasets"]), "dataset": dataset}
        write_json(_run_dir(cfg, experiment_id, model_name, cloud["datasets"][0]).parent / "progress.json", state)
        if progress:
            progress(state)


def replay_stage(cfg: Config, experiment_id: str, model_name: str, model, tokenizer, family, mcfg,
                 progress: Callable[[dict], None] | None = None) -> None:
    cloud = cfg["cloud"]
    for dataset_index, dataset in enumerate(cloud["datasets"]):
        run_dir = _run_dir(cfg, experiment_id, model_name, dataset)
        records = read_jsonl(run_dir / "generations.jsonl")

        def row_progress(done, total):
            state = {"stage": "activation_replay", "dataset": dataset,
                     "datasets_complete": dataset_index, "datasets_total": len(cloud["datasets"]),
                     "rows_complete": done, "rows_total": total}
            write_json(run_dir / "progress.json", state)
            if progress:
                progress(state)

        result = extract_shards(model, tokenizer, family, records, run_dir / "replay",
                                int(cloud["replay_chunk_rows"]), int(cfg["replay"]["logit_chunk"]),
                                int(mcfg["expected_text_layers"]), int(mcfg["expected_hidden_size"]), row_progress)
        if result["status"] != "ok":
            raise RuntimeError(f"{dataset} replay failed: {result}")
        materialize(run_dir / "replay", records)


def release_model(model) -> None:
    if hasattr(model, "close"):
        model.close()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

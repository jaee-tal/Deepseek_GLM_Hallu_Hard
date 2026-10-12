"""vLLM subject backend for the local Gemma 4 experiment.

The published Gemma 4 MoE checkpoint cannot be quantized into 24 GB by
Transformers/bitsandbytes because its packed expert tensors are not Linear
modules.  The pinned ExpertsINT4-FP8 derivative preserves the same model and
is loaded with vLLM's compressed-tensors kernels.  vLLM's hidden-state
connector exports the same representation used elsewhere in this repository:
embedding output plus every text-block output, mean-pooled over answer tokens.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def _patch_flashinfer_python311() -> None:
    """Make flashinfer 0.6's typed ``array.array`` annotation lazy on Py3.11.

    vLLM imports this module even when its sampler is disabled.  Python 3.12
    accepts the annotation, while the local Python 3.11 runtime needs postponed
    evaluation.  The patch is idempotent and is applied before importing vLLM.
    """
    if sys.version_info >= (3, 12):
        return
    spec = importlib.util.find_spec("flashinfer")
    if spec is None or not spec.submodule_search_locations:
        return
    path = Path(next(iter(spec.submodule_search_locations))) / "comm" / "fd_exchange.py"
    if not path.exists():
        return
    source = path.read_text(encoding="utf-8")
    future = "from __future__ import annotations"
    if future in source:
        return
    # The installed file starts with its license comments and a one-line
    # module docstring.  A future import must follow that docstring.
    marker = '"""Exchange file descriptors between processes."""\n'
    if marker not in source:
        raise RuntimeError(f"cannot safely patch unexpected flashinfer source: {path}")
    path.write_text(source.replace(marker, marker + future + "\n", 1), encoding="utf-8")


def _patch_vllm_hidden_state_pages() -> None:
    """Allow hidden-state cache pages wider than an attention-cache page.

    vLLM 0.27.1 aligns the extraction cache down to the attention page.  That
    works for a few selected layers but asserts when all Gemma layers exceed a
    single attention page.  Pad every cache group to the next whole attention
    page so all 31 residual streams can be exported in one forward pass.
    """
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        return
    path = (Path(next(iter(spec.submodule_search_locations))) / "v1" / "core" /
            "kv_cache_utils.py")
    if not path.exists():
        return
    source = path.read_text(encoding="utf-8")
    marker = "# cross-probe: widen all groups for multi-layer hidden-state export"
    if marker in source:
        return
    old = '''    # Add hidden-state layers back with page aligned to the common page.
    if hidden_specs:
        common_page = get_uniform_page_size([g.kv_cache_spec for g in groups])
        for name, spec in hidden_specs.items():
            per_token = spec.num_kv_heads * spec.head_size * get_dtype_size(spec.dtype)
            new_bs = max(common_page // per_token, 1)
            aligned = replace(spec, block_size=new_bs, page_size_padded=common_page)
            groups.append(KVCacheGroupSpec([name], aligned))
'''
    new = '''    # Add hidden-state layers back with page aligned to the common page.
    if hidden_specs:
        common_page = get_uniform_page_size([g.kv_cache_spec for g in groups])
        required = max(
            spec.num_kv_heads * spec.head_size * get_dtype_size(spec.dtype)
            for spec in hidden_specs.values()
        )
        if required > common_page:
            # cross-probe: widen all groups for multi-layer hidden-state export
            common_page = cdiv(required, common_page) * common_page
            groups = [
                KVCacheGroupSpec(
                    group.layer_names,
                    replace(group.kv_cache_spec, page_size_padded=common_page),
                )
                for group in groups
            ]
        for name, spec in hidden_specs.items():
            per_token = spec.num_kv_heads * spec.head_size * get_dtype_size(spec.dtype)
            new_bs = max(common_page // per_token, 1)
            aligned = replace(spec, block_size=new_bs, page_size_padded=common_page)
            groups.append(KVCacheGroupSpec([name], aligned))
'''
    if old not in source:
        raise RuntimeError(f"cannot safely patch unexpected vLLM cache source: {path}")
    path.write_text(source.replace(old, new, 1), encoding="utf-8")


def _gemma4_hf_overrides(config):
    """Supply vLLM with Gemma 4's heterogeneous global-attention dimensions."""
    config.allow_global_per_layer_attribute_access = True
    text = config.text_config if hasattr(config, "text_config") else config
    text.allow_global_per_layer_attribute_access = True
    text.global_head_dim = 512
    text.num_global_key_value_heads = 2
    return config


class VllmSubject:
    """Generate responses and export answer-pooled residual activations."""

    cross_probe_backend = "vllm"

    def __init__(self, model_cfg: dict[str, Any], *, extract_hidden_states: bool = False):
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        _patch_flashinfer_python311()
        _patch_vllm_hidden_state_pages()

        from vllm import LLM

        self.model_cfg = dict(model_cfg)
        self.extract_hidden_states = bool(extract_hidden_states)
        self.n_hidden_states = int(model_cfg["expected_text_layers"]) + 1
        self.hidden_size = int(model_cfg["expected_hidden_size"])
        self.layer_ids = list(range(self.n_hidden_states))
        self.hidden_dir: Path | None = None

        kwargs: dict[str, Any] = {
            "model": model_cfg["checkpoint"],
            "revision": model_cfg.get("revision"),
            "dtype": model_cfg.get("dtype", "bfloat16"),
            "max_model_len": int(model_cfg.get(
                "vllm_replay_max_model_len" if self.extract_hidden_states else "vllm_max_model_len",
                768 if self.extract_hidden_states else 4096,
            )),
            "kv_cache_memory_bytes": int(model_cfg.get("vllm_kv_cache_bytes", 1_600_000_000)),
            "max_num_seqs": int(model_cfg.get("vllm_max_num_seqs", 8)),
            "enable_chunked_prefill": False,
            # Repeated teacher-forced requests must execute every token so the
            # hidden-state connector never reads residuals from a prior request.
            "enable_prefix_caching": not self.extract_hidden_states,
            "enforce_eager": bool(model_cfg.get("vllm_enforce_eager", True)),
            "limit_mm_per_prompt": {"image": 0, "video": 0},
            "hf_overrides": _gemma4_hf_overrides,
        }
        if self.extract_hidden_states:
            root = Path(model_cfg.get("vllm_hidden_state_dir", "/dev/shm"))
            root.mkdir(parents=True, exist_ok=True)
            self.hidden_dir = Path(tempfile.mkdtemp(prefix="cross-probe-gemma4-", dir=root))
            kwargs["speculative_config"] = {
                "method": "extract_hidden_states",
                "num_speculative_tokens": 1,
                "draft_model_config": {
                    "hf_config": {"eagle_aux_hidden_state_layer_ids": self.layer_ids},
                },
            }
            kwargs["kv_transfer_config"] = {
                "kv_connector": "ExampleHiddenStatesConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "shared_storage_path": str(self.hidden_dir),
                    "num_writer_threads": int(model_cfg.get("vllm_hidden_writer_threads", 4)),
                    "use_synchronization_lock": True,
                },
            }
        log.info("loading vLLM subject %s (hidden extraction=%s)", model_cfg["checkpoint"], self.extract_hidden_states)
        self.llm = LLM(**{key: value for key, value in kwargs.items() if value is not None})
        self.tokenizer = self.llm.get_tokenizer()

    def quantization_info(self) -> dict[str, Any]:
        import torch

        allocated = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else float("nan")
        return {
            "backend": "vllm",
            "quantization_requested": self.model_cfg.get("quantization"),
            "quant_method": "compressed-tensors",
            "weight_format": "experts_int4_gptq_group64_plus_fp8_linears",
            "memory_allocated_gb": round(float(allocated), 2),
            "all_weights_on_gpu": True,
            "base_checkpoint": self.model_cfg.get("base_checkpoint"),
            "base_revision": self.model_cfg.get("base_revision"),
        }

    @staticmethod
    def _config_hash(gen_cfg: dict[str, Any]) -> str:
        payload = {key: gen_cfg.get(key) for key in
                   ("do_sample", "temperature", "top_p", "top_k", "max_new_tokens")}
        payload["backend"] = "vllm"
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def _cleanup_generation_hidden_states(self, output) -> None:
        """Generation stages do not retain connector output from smoke calls."""
        params = getattr(output, "kv_transfer_params", None) or {}
        path = params.get("hidden_states_path")
        if not path:
            return
        from vllm.distributed.kv_transfer.kv_connector.v1 import example_hidden_states_connector

        # Wait for the asynchronous writer before removing the file.
        example_hidden_states_connector.load_hidden_states(path)
        example_hidden_states_connector.cleanup_hidden_states(path)

    def generate_all(self, tokenizer, family, jobs: list[dict[str, Any]], gen_cfg: dict[str, Any],
                     out_path: Path, model_name: str, domain: str, batch_size: int) -> None:
        """vLLM equivalent of vendor.generation.generate_all with stable resume seeds."""
        from vllm import SamplingParams

        from ..vendor.generation import (batch_seed, cut_at_eos, existing_conversation_ids,
                                         _sorted_groups)
        from ..vendor.io_utils import append_jsonl, sha256_text

        done = existing_conversation_ids(out_path)
        todo = [job for job in jobs if job["conversation_id"] not in done]
        log.info("vLLM generation: %d jobs, %d already done, %d to do", len(jobs), len(done), len(todo))
        if not todo:
            return
        flat = [row for group in _sorted_groups(jobs) for row in group]
        batches = [flat[i:i + batch_size] for i in range(0, len(flat), batch_size)]
        work = [(idx, [row for row in batch if row["conversation_id"] not in done])
                for idx, batch in enumerate(batches)]
        work = [(idx, batch) for idx, batch in work if batch]
        base_seed = int(gen_cfg["base_seed"])
        gc_hash = self._config_hash(gen_cfg)
        n_done = 0
        for work_idx, (batch_index, batch) in enumerate(work):
            started = time.time()
            members = [[row["question_id"], row["sample_id"]] for row in batch]
            prompts, params, seeds = [], [], []
            for position, row in enumerate(batch):
                encoded = tokenizer(row["prompt_text"], add_special_tokens=False)["input_ids"]
                if list(encoded) != row["prompt_token_ids"]:
                    raise RuntimeError(f"prompt alignment failure for {row['conversation_id']}")
                prompts.append({"prompt_token_ids": row["prompt_token_ids"]})
                seed = batch_seed(base_seed, batch_index * 1000 + position)
                seeds.append(seed)
                params.append(SamplingParams(
                    n=1,
                    max_tokens=int(gen_cfg["max_new_tokens"]),
                    temperature=float(gen_cfg["temperature"]) if gen_cfg.get("do_sample", True) else 0.0,
                    top_p=float(gen_cfg["top_p"]),
                    top_k=int(gen_cfg.get("top_k", 0)),
                    repetition_penalty=1.0,
                    seed=seed,
                    stop_token_ids=list(family.eos_ids),
                    skip_special_tokens=False,
                ))
            outputs = self.llm.generate(prompts, params, use_tqdm=False)
            records = []
            for position, (row, request) in enumerate(zip(batch, outputs)):
                completion = request.outputs[0]
                generated, cut_reason = cut_at_eos(
                    list(map(int, completion.token_ids)), family.eos_ids, family.pad_id
                )
                stop_reason = "length" if completion.finish_reason == "length" else cut_reason
                span = family.answer_span(generated)
                n_prompt = len(row["prompt_token_ids"])
                response_text = tokenizer.decode(generated[span.start:span.end], skip_special_tokens=True)
                full_text = tokenizer.decode(generated, skip_special_tokens=False)
                records.append({
                    "model": model_name, "domain": domain,
                    "question_id": row["question_id"], "sample_id": row["sample_id"],
                    "conversation_id": row["conversation_id"], "condition": row.get("condition"),
                    "template_id": row.get("template_id"), "framing": row.get("framing", "none"),
                    "instruction_type": row.get("instruction_type"),
                    "held_out_template": row.get("held_out_template"), "split": row.get("split"),
                    "seed": seeds[position], "seed_scheme": "per_request_vllm_v1",
                    "base_seed": base_seed, "batch_index": batch_index, "sub_batch_index": 0,
                    "position_in_batch": position, "batch_members": members,
                    "prompt_text": row["prompt_text"], "prompt_token_ids": row["prompt_token_ids"],
                    "generated_token_ids": generated, "n_generated": len(generated),
                    "answer_start": span.start, "answer_end": span.end,
                    "response_token_start": n_prompt + span.start,
                    "response_token_end": n_prompt + span.end,
                    "n_answer_tokens": span.end - span.start, "response_text": response_text,
                    "full_generation_text": full_text, "stop_reason": stop_reason,
                    "truncated": stop_reason == "length", "exclude_reason": span.exclude_reason,
                    "span_flags": span.flags, "response_sha256": sha256_text(response_text),
                    "temperature": float(gen_cfg["temperature"]), "top_p": float(gen_cfg["top_p"]),
                    "top_k": int(gen_cfg.get("top_k", 0)),
                    "max_new_tokens": int(gen_cfg["max_new_tokens"]),
                    "generation_config_hash": gc_hash,
                })
                if self.extract_hidden_states:
                    self._cleanup_generation_hidden_states(request)
            append_jsonl(out_path, records)
            n_done += len(records)
            log.info("vLLM batch %d/%d (plan batch %d; %d rows) in %.1fs — %d/%d done",
                     work_idx + 1, len(work), batch_index, len(records), time.time() - started,
                     n_done, len(todo))

    @staticmethod
    def _actual_token_logprobs(request, token_ids: list[int], start: int, end: int) -> np.ndarray:
        values = []
        prompt_logprobs = getattr(request, "prompt_logprobs", None) or []
        for pos in range(start, end):
            entry = prompt_logprobs[pos] if pos < len(prompt_logprobs) else None
            item = entry.get(token_ids[pos]) if entry else None
            values.append(float(item.logprob) if item is not None else float("nan"))
        return np.asarray(values, dtype=np.float32)

    def replay_batch(self, records: list[dict[str, Any]]) -> dict[str, tuple[np.ndarray, dict, np.ndarray, np.ndarray]]:
        """Teacher-force stored sequences and pool all layer outputs over answers."""
        if not self.extract_hidden_states:
            raise RuntimeError("vLLM subject was loaded without hidden-state extraction")
        from vllm import SamplingParams
        from vllm.distributed.kv_transfer.kv_connector.v1 import example_hidden_states_connector

        prompts, params, full_by_id = [], [], {}
        for record in records:
            full = list(map(int, record["prompt_token_ids"] + record["generated_token_ids"]))
            replay_limit = int(self.model_cfg.get("vllm_replay_max_model_len", 768))
            if len(full) >= replay_limit:
                raise RuntimeError(
                    f"{record['conversation_id']} has {len(full)} tokens, exceeding the configured "
                    f"vLLM replay context ({replay_limit})"
                )
            full_by_id[record["conversation_id"]] = full
            prompts.append({"prompt_token_ids": full})
            params.append(SamplingParams(
                max_tokens=1, temperature=0.0, prompt_logprobs=1,
                extra_args={"kv_transfer_params": {"include_output_tokens": False}},
            ))
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        result = {}
        for record, request in zip(records, outputs):
            cid = record["conversation_id"]
            path = request.kv_transfer_params["hidden_states_path"]
            try:
                obj = example_hidden_states_connector.load_hidden_states(path)
                token_ids = list(map(int, obj["token_ids"].tolist()))
                full = full_by_id[cid]
                if token_ids != full:
                    raise RuntimeError(f"hidden-state token alignment failure for {cid}")
                hidden = obj["hidden_states"]
                expected = (len(full), self.n_hidden_states, self.hidden_size)
                if tuple(hidden.shape) != expected:
                    raise RuntimeError(f"hidden-state shape for {cid} is {tuple(hidden.shape)}, expected {expected}")
                start, end = int(record["response_token_start"]), int(record["response_token_end"])
                pooled = hidden[start:end].float().mean(dim=0).numpy()
                lp = self._actual_token_logprobs(request, full, start, end)
                entropy = np.full(end - start, np.nan, dtype=np.float32)
                finite = lp[np.isfinite(lp)]
                stats = {
                    "n_answer_tokens": end - start,
                    "sum_logprob": float(finite.sum()) if finite.size else float("nan"),
                    "mean_logprob": float(finite.mean()) if finite.size else float("nan"),
                    "min_logprob": float(finite.min()) if finite.size else float("nan"),
                    # vLLM does not expose the full-vocabulary prompt logits.
                    "mean_entropy": float("nan"), "max_entropy": float("nan"),
                    "first_token_entropy": float("nan"), "mean_top1_prob": float("nan"),
                    "statistics_backend": "vllm_prompt_logprobs_top1",
                }
                result[cid] = (pooled, stats, lp, entropy)
            finally:
                example_hidden_states_connector.cleanup_hidden_states(path)
        return result

    def close(self) -> None:
        if self.hidden_dir:
            shutil.rmtree(self.hidden_dir, ignore_errors=True)
            self.hidden_dir = None

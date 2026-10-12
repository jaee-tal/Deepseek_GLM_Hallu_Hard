"""Model/tokenizer loading with version-tolerant kwargs."""
from __future__ import annotations

import os
from typing import Any

import torch

from pathlib import Path


def setup_hf_env(hf_home: str | Path) -> None:
    hf_home = Path(hf_home)
    hf_home.mkdir(parents=True, exist_ok=True)
    # model + raw dataset files are read from the shared hub cache; processed datasets go to this project's own cache
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    own = Path(__file__).resolve().parents[2] / "data" / "hf_datasets_cache"
    own.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_DATASETS_CACHE", str(own))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_model_and_tokenizer(model_cfg: dict[str, Any], device: str = "cuda"):
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt = model_cfg["checkpoint"]
    if model_cfg["family"] == "llama31" and not os.environ.get("HF_TOKEN"):
        raise SystemExit("Llama-3.1 is gated: put HF_TOKEN=<token with access> in .env before running this model.")

    revision = model_cfg.get("revision")
    common = {"revision": revision, "trust_remote_code": bool(model_cfg.get("trust_remote_code", False))}
    common = {k: v for k, v in common.items() if v is not None}
    if model_cfg.get("processor"):
        processor = transformers.AutoProcessor.from_pretrained(ckpt, **common)
        tok = getattr(processor, "tokenizer", processor)
        # Keep the processor reachable for future multimodal extensions while the
        # present experiment remains deliberately text-only.
        tok._cross_probe_processor = processor
    else:
        tok = AutoTokenizer.from_pretrained(ckpt, **common)
    dtype = getattr(torch, model_cfg.get("dtype", "bfloat16"))
    configured_map = model_cfg.get("device_map")
    device_map = configured_map if configured_map is not None else ({"": 0} if device == "cuda" else None)
    kwargs: dict[str, Any] = {"device_map": device_map, "low_cpu_mem_usage": True, **common}
    if model_cfg.get("attn_implementation"):
        kwargs["attn_implementation"] = model_cfg["attn_implementation"]
    quant = model_cfg.get("quantization", "none")
    if quant == "bnb_nf4":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True
        )
    elif quant in ("mxfp4_native", "bnb_prequantized", "fp8_native"):
        # The checkpoint carries its own quantization config; stacking another one is a hard error in transformers.
        # MXFP4 needs the `kernels` package, otherwise transformers silently dequantizes to bf16 (checked after load).
        pass
    elif quant != "none":
        raise ValueError(f"unknown quantization {quant}")
    loader = model_cfg.get("loader", "causal_lm")
    if loader == "causal_lm":
        model_cls = AutoModelForCausalLM
    elif loader == "multimodal_lm":
        model_cls = getattr(transformers, "AutoModelForMultimodalLM", None) or getattr(transformers, "AutoModelForImageTextToText", None)
        if model_cls is None:
            raise RuntimeError("this transformers build has no multimodal auto-model class; install the version pinned in requirements.txt")
    else:
        raise ValueError(f"unknown model loader {loader!r}")
    try:
        model = model_cls.from_pretrained(ckpt, dtype=dtype, **kwargs)
    except TypeError:
        model = model_cls.from_pretrained(ckpt, torch_dtype=dtype, **kwargs)
    model.eval()
    return model, tok


class QuantizationCheckFailed(RuntimeError):
    """The model did not load in the requested quantized form (e.g. MXFP4 silently dequantized to bf16)."""


def quantization_info(model) -> dict[str, Any]:
    qc = getattr(model.config, "quantization_config", None)
    method = (qc.get("quant_method") if isinstance(qc, dict) else getattr(qc, "quant_method", None)) if qc is not None else None
    method = getattr(method, "value", method)
    try:
        gb = float(model.get_memory_footprint()) / 1e9
    except Exception:  # noqa: BLE001
        gb = float(torch.cuda.memory_allocated()) / 1e9 if torch.cuda.is_available() else float("nan")
    return {"quant_method": None if method is None else str(method), "memory_footprint_gb": round(gb, 2)}


def check_quantization(model, model_cfg: dict[str, Any]) -> dict[str, Any]:
    """Raise QuantizationCheckFailed unless the model really loaded in the requested quantized form and size."""
    quant = model_cfg.get("quantization", "none")
    info = {"quantization_requested": quant, **quantization_info(model)}
    limit = model_cfg.get("max_gpu_gb_after_load")
    expected = {"mxfp4_native": ("mxfp4",), "fp8_native": ("fp8", "fbgemm_fp8"),
                "bnb_prequantized": ("bitsandbytes", "bitsandbytes_4bit"),
                "bnb_nf4": ("bitsandbytes", "bitsandbytes_4bit")}.get(quant)
    if expected and info["quant_method"] not in expected:
        hint = " (is the `kernels` package installed? without it transformers dequantizes MXFP4 to bf16)" if quant == "mxfp4_native" else ""
        raise QuantizationCheckFailed(f"{model_cfg['checkpoint']}: quant_method={info['quant_method']!r}, expected one of {expected}{hint}")
    if limit is not None and info["memory_footprint_gb"] > float(limit):
        raise QuantizationCheckFailed(f"{model_cfg['checkpoint']}: {info['memory_footprint_gb']} GB after load > limit {limit} GB")
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        info["device_map"] = {str(k): str(v) for k, v in device_map.items()}
    if model_cfg.get("require_all_on_gpu"):
        if not isinstance(device_map, dict) or not device_map:
            raise QuantizationCheckFailed(f"{model_cfg['checkpoint']}: no Accelerate device map available to verify GPU residency")
        offloaded = {str(k): str(v) for k, v in device_map.items()
                     if str(v).casefold() in {"cpu", "disk", "meta"}}
        if offloaded:
            preview = dict(list(offloaded.items())[:8])
            raise QuantizationCheckFailed(
                f"{model_cfg['checkpoint']}: layers were offloaded from the GPU: {preview}; "
                "the local experiment requires the complete quantized model on GPU"
            )
    return info


def model_commit_sha(checkpoint: str, revision: str | None = None) -> str | None:
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(checkpoint, revision=revision).sha
    except Exception:
        return None

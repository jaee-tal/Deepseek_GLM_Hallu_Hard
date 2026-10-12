"""Shared CLI helpers. Every stage is addressed by (--model, --dataset, --tag)."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cross_probe.config import Config, load_config  # noqa: E402
from cross_probe.vendor.io_utils import load_dotenv  # noqa: E402

QUANTIZATION_EXIT_CODE = 3

log = logging.getLogger("cross_probe")


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", default=None)
    p.add_argument("--model", default=None, help="key in configs/experiment.yaml `models:` (default: default_model)")
    p.add_argument("--dataset", default=None, help="key in configs/experiment.yaml `datasets:` (default: default_dataset)")
    p.add_argument("--tag", default="pilot", help="run tag (smoke | pilot | ...)")
    p.add_argument("--no-gate", action="store_true", help="skip the free-GPU gate")
    return p


def setup(args) -> Config:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv()
    return load_config(args.config)


def names(cfg: Config, args) -> tuple[str, str]:
    return cfg.model_name(getattr(args, "model", None)), cfg.dataset_name(getattr(args, "dataset", None))


def run_dir_for(cfg: Config, args) -> Path:
    d = cfg.run_dir(args.tag, getattr(args, "model", None), getattr(args, "dataset", None))
    d.mkdir(parents=True, exist_ok=True)
    (d / "logs").mkdir(exist_ok=True)
    return d


def gate(cfg: Config, args, require_gpu: bool = True) -> None:
    if getattr(args, "no_gate", False):
        return
    from cross_probe.gpu import wait_for_gate

    wait_for_gate(cfg["gate"], require_gpu=require_gpu)


def load_subject(cfg: Config, args):
    """Load the subject model + tokenizer + family and verify it really loaded in the requested quantized form.

    A failed quantization check exits with QUANTIZATION_EXIT_CODE naming the configured `fallback_model`, so that
    the run directory describes the model that was actually used."""
    from cross_probe.vendor.families import make_family
    from cross_probe.vendor.modeling import QuantizationCheckFailed, check_quantization, load_model_and_tokenizer

    name = cfg.model_name(getattr(args, "model", None))
    mcfg = cfg.model(name)
    model, tok = load_model_and_tokenizer(mcfg)
    try:
        info = check_quantization(model, mcfg)
    except QuantizationCheckFailed as e:
        fb = mcfg.get("fallback_model")
        log.error("quantization check failed for %s: %s", name, e)
        if fb:
            log.error("re-run this stage with --model %s (the configured fallback)", fb)
            print(f"FALLBACK_MODEL={fb}", flush=True)
        raise SystemExit(QUANTIZATION_EXIT_CODE)
    info["model_name"] = name
    log.info("loaded %s: %s", name, info)
    family = make_family(mcfg["family"], tok, mcfg.get("template_kwargs", {}), mcfg)
    return model, tok, family, name, mcfg, info

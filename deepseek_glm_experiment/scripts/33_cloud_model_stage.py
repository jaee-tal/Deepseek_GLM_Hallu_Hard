"""Run one GPU stage while keeping the large model loaded across datasets."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from _common import base_parser, setup
from cross_probe.cloud.model_stages import (generate_stage, load_subject, model_smoke,
                                            release_model, replay_stage)
from cross_probe.config import CLOUD_CONFIG
from cross_probe.vendor.io_utils import PROJECT_ROOT


def main() -> None:
    parser: argparse.ArgumentParser = base_parser("cloud GPU stage")
    parser.set_defaults(config=str(CLOUD_CONFIG))
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--stage", required=True,
                        choices=("organic_generation", "inorganic_generation", "activation_replay", "preflight"))
    parser.add_argument("--skip-model-smoke", action="store_true")
    args = parser.parse_args()
    cfg = setup(args)
    model_name = args.model or cfg.model_name()
    if model_name not in cfg["cloud"]["models"]:
        parser.error(f"--model must be one of {cfg['cloud']['models']}")
    control = PROJECT_ROOT / "runs" / cfg["cloud"]["experiment_namespace"] / args.experiment_id / model_name
    control.mkdir(parents=True, exist_ok=True)
    need_hidden_states = args.stage in ("activation_replay", "preflight")
    model, tokenizer, family, mcfg, qinfo = load_subject(
        cfg, model_name, extract_hidden_states=need_hidden_states
    )
    try:
        if args.stage == "preflight" or not args.skip_model_smoke:
            print("model smoke:", model_smoke(cfg, model_name, control, model, tokenizer, family, mcfg))
        if args.stage == "organic_generation":
            generate_stage(cfg, args.experiment_id, model_name, "organic", model, tokenizer, family, mcfg, qinfo)
        elif args.stage == "inorganic_generation":
            generate_stage(cfg, args.experiment_id, model_name, "inorganic", model, tokenizer, family, mcfg, qinfo)
        elif args.stage == "activation_replay":
            replay_stage(cfg, args.experiment_id, model_name, model, tokenizer, family, mcfg)
    finally:
        release_model(model)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()

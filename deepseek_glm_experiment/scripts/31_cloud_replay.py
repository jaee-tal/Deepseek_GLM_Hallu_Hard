"""Chunked replay/extraction for cloud_v3 runs."""
from __future__ import annotations

import argparse
import logging

from _common import base_parser, gate, load_subject, run_dir_for, setup
from cross_probe.cloud.replay import extract_shards, materialize
from cross_probe.config import CLOUD_CONFIG
from cross_probe.vendor.io_utils import read_jsonl, write_json
from cross_probe.vendor.modeling import setup_hf_env


def main() -> None:
    parser: argparse.ArgumentParser = base_parser("immutable-shard cloud replay")
    parser.set_defaults(config=str(CLOUD_CONFIG))
    parser.add_argument("--materialize-only", action="store_true")
    parser.add_argument("--allow-alignment-failures", action="store_true")
    args = parser.parse_args()
    cfg = setup(args)
    run_dir = run_dir_for(cfg, args)
    out = run_dir / "replay"
    records = read_jsonl(run_dir / "generations.jsonl")
    if not args.materialize_only:
        setup_hf_env(cfg.hf_home())
        gate(cfg, args, require_gpu=True)
        model, tokenizer, family, model_name, mcfg, _ = load_subject(cfg, args)
        cloud = cfg["cloud"]

        def progress(done, total):
            write_json(run_dir / "progress.json", {"stage": "activation_replay", "completed": done, "total": total})

        result = extract_shards(model, tokenizer, family, records, out,
                                int(cloud["replay_chunk_rows"]), int(cfg["replay"]["logit_chunk"]),
                                int(mcfg["expected_text_layers"]), int(mcfg["expected_hidden_size"]), progress)
        if result["status"] != "ok" and not args.allow_alignment_failures:
            raise SystemExit(f"replay alignment failures: {result}")
    print("materialized:", materialize(out, records))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

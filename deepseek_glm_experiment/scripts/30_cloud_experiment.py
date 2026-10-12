"""Portable entry point for DeepSeek, GLM, and Gemma cloud experiments."""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys

from _common import setup
from cross_probe.cloud.pipeline import CloudPipeline, STAGES
from cross_probe.config import CLOUD_CONFIG


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("validate", "preflight", "run", "stage", "checkpoint", "restore", "status"))
    p.add_argument("--config", default=str(CLOUD_CONFIG))
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--artifact-uri", default=None, help="file://... or gs://bucket/prefix; overrides config/env")
    p.add_argument("--stage", choices=STAGES)
    p.add_argument("--start", choices=STAGES)
    p.add_argument("--stop", choices=STAGES)
    p.add_argument("--force", action="store_true")
    p.add_argument("--offline", action="store_true", help="skip credentials/network/CUDA checks in validate")
    return p


def main() -> None:
    args = parser().parse_args()
    if args.offline and args.command in ("run", "stage"):
        raise SystemExit("--offline is only for validate/preflight inspection; experiment stages require monitoring and credential checks")
    cfg = setup(args)
    pipeline = CloudPipeline(cfg, args.experiment_id, args.model, args.artifact_uri)
    if args.command in ("validate", "preflight"):
        print(json.dumps(pipeline.static_preflight(online=not args.offline), indent=2))
        if args.command == "preflight" and not args.offline:
            command = [sys.executable, "scripts/33_cloud_model_stage.py", "--config", str(cfg.path),
                       "--model", args.model, "--experiment-id", args.experiment_id,
                       "--stage", "preflight", "--no-gate"]
            subprocess.check_call(command)
            pipeline.checkpoints.checkpoint("preflight", {"status": "complete"})
            pipeline.monitoring_preflight()
        return
    if args.command == "status":
        print(json.dumps(pipeline.status(), indent=2))
    elif args.command == "checkpoint":
        if not args.stage:
            raise SystemExit("--stage is required")
        print(json.dumps(pipeline.checkpoints.checkpoint(args.stage), indent=2))
    elif args.command == "restore":
        restored = pipeline.restore(args.stage) if args.stage else pipeline.restore_latest()
        print(json.dumps(restored, indent=2))
    elif args.command == "stage":
        if not args.stage:
            raise SystemExit("--stage is required")
        pipeline.static_preflight(online=not args.offline)
        pipeline.run_stage(args.stage, force=args.force)
    elif args.command == "run":
        pipeline.static_preflight(online=not args.offline)
        pipeline.run(args.start, args.stop)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()

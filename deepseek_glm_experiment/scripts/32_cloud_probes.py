"""Fit or report all four pooled cloud_v3 probe arms."""
from __future__ import annotations

import argparse

from _common import base_parser, setup
from cross_probe.cloud.probes import run_experiment
from cross_probe.cloud.report import report
from cross_probe.config import CLOUD_CONFIG
from cross_probe.pooled_transfer import PooledData
from cross_probe.vendor.io_utils import PROJECT_ROOT


def main() -> None:
    parser: argparse.ArgumentParser = base_parser("cloud_v3 pooled probe experiment")
    parser.set_defaults(config=str(CLOUD_CONFIG))
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--phase", choices=("fit", "report", "all"), default="all")
    parser.add_argument("--layers", help="optional comma-separated layer indices for smoke/resume")
    args = parser.parse_args()
    cfg = setup(args)
    model = args.model or cfg.model_name()
    cloud = cfg["cloud"]
    if model not in cloud["models"]:
        parser.error(f"--model must be one of {cloud['models']}")
    root = PROJECT_ROOT / "runs" / "summary" / cloud["experiment_namespace"] / args.experiment_id / model
    output = PROJECT_ROOT / "results" / cloud["experiment_namespace"] / args.experiment_id / model
    root.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    if args.phase in ("fit", "all"):
        data = PooledData.load(cfg, tag=args.experiment_id, model=model)
        run_experiment(data, root, tuple(cloud["probes"]), tuple(cloud["registers"]),
                       tuple(cloud["arms"]), tuple(map(int, cloud["seeds"])),
                       float(cloud["organic_fraction"]), float(cloud["organic_mass"]), layers)
    if args.phase in ("report", "all"):
        summary = report(root, output, tuple(cloud["probes"]))
        print(f"report: {len(summary)} aggregate rows -> {output}")


if __name__ == "__main__":
    main()

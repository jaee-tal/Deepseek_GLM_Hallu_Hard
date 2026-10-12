"""Stage orchestration, checkpoint lifecycle, restore, and stall detection."""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch
from packaging.version import Version
import transformers

from ..config import Config, questions_path, split_path
from ..judge import check_credit
from ..vendor.io_utils import PROJECT_ROOT, read_json, sha256_file, write_json
from ..vendor.modeling import model_commit_sha
from .artifacts import ArtifactStore
from .checkpoint import CheckpointManager
from .monitoring import RunMonitor

log = logging.getLogger(__name__)

STAGES = ("prepare", "organic_generation", "organic_judging", "eligibility",
          "inorganic_generation", "inorganic_judging", "activation_replay",
          "dataset_build", "probe_fit", "report")


class CloudPipeline:
    def __init__(self, cfg: Config, experiment_id: str, model: str, artifact_uri: str | None = None):
        self.cfg, self.experiment_id, self.model = cfg, experiment_id, model
        cloud = cfg["cloud"]
        if model not in cloud["models"]:
            raise ValueError(f"model must be one of {cloud['models']}")
        self.namespace = f"cross-probe/{cloud['experiment_namespace']}/{experiment_id}/{model}"
        self.control = PROJECT_ROOT / "runs" / cloud["experiment_namespace"] / experiment_id / model
        self.summary = PROJECT_ROOT / "runs" / "summary" / cloud["experiment_namespace"] / experiment_id / model
        self.results = PROJECT_ROOT / "results" / cloud["experiment_namespace"] / experiment_id / model
        self.control.mkdir(parents=True, exist_ok=True)
        self.local_roots = {"control": self.control, "summary": self.summary, "results": self.results}
        for dataset in cloud["datasets"]:
            self.local_roots[f"run_{dataset}"] = cfg.run_dir(experiment_id, model, dataset)
        store = ArtifactStore.from_uri(artifact_uri or cloud["artifact_uri"])
        mcfg = cfg.model(model)
        self.monitor: RunMonitor | None = None
        self.checkpoints = CheckpointManager(
            store, self.namespace, self.local_roots, cfg.hash,
            model_revision=mcfg.get("revision") or model_commit_sha(mcfg["checkpoint"]),
            container_digest=os.environ.get("CROSS_PROBE_CONTAINER_DIGEST"),
            on_checkpoint=lambda manifest: self.monitor.checkpoint(manifest) if self.monitor else None,
        )

    def marker(self, stage: str) -> Path:
        return self.control / "stages" / f"{stage}.json"

    def is_complete(self, stage: str) -> bool:
        if not self.marker(stage).exists():
            return False
        marker = read_json(self.marker(stage))
        return marker.get("status") == "complete" and marker.get("config_hash") == self.cfg.hash

    def static_preflight(self, online: bool = True) -> dict[str, Any]:
        cloud, mcfg = self.cfg["cloud"], self.cfg.model(self.model)
        if not self.experiment_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in self.experiment_id):
            raise ValueError("experiment ID may contain only letters, digits, '-' and '_'")
        inputs = {}
        for dataset in cloud["datasets"]:
            qpath, spath = questions_path(self.cfg, dataset), split_path(self.cfg, dataset)
            if not qpath.exists() or not spath.exists():
                raise FileNotFoundError(f"missing fixed inputs for {dataset}; run 01_prepare_questions.py first")
            split = read_json(spath)
            ids = [qid for name in ("train", "validation", "test") for qid in split[name]]
            if len(ids) != len(set(ids)):
                raise RuntimeError(f"question split overlap in {dataset}")
            inputs[dataset] = {"questions_sha256": sha256_file(qpath), "split_sha256": sha256_file(spath),
                               "counts": {name: len(split[name]) for name in ("train", "validation", "test")}}
        if online:
            installed_transformers = transformers.__version__
            if Version(installed_transformers) < Version(str(mcfg["minimum_transformers"])):
                raise RuntimeError(f"{self.model} requires transformers>={mcfg['minimum_transformers']}; "
                                   f"this environment has {installed_transformers}. Build Dockerfile.cloud or update the environment.")
            if not os.environ.get("OPENROUTER_API_KEY"):
                raise RuntimeError("OPENROUTER_API_KEY is required for GPT-mini judging")
            credit = check_credit(min_balance=1.0)
            wcfg = cloud.get("wandb", {})
            if (wcfg.get("enabled") and str(wcfg.get("mode", "online")).casefold() != "offline"
                    and not os.environ.get("WANDB_API_KEY")):
                raise RuntimeError("WANDB_API_KEY is required while cloud.wandb is online")
            if not os.environ.get("CROSS_PROBE_CONTAINER_DIGEST"):
                raise RuntimeError("CROSS_PROBE_CONTAINER_DIGEST is required for a provenance-pinned cloud run")
        else:
            credit = {"skipped": True}
            installed_transformers = transformers.__version__
        if torch.cuda.is_available():
            gpu = {"count": torch.cuda.device_count(),
                   "total_memory_gb": sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) / 1e9,
                   "names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
        else:
            gpu = {"count": 0, "total_memory_gb": 0, "names": []}
            if online:
                raise RuntimeError("CUDA GPUs are required for the subject-model stages")
        # Verify the artifact backend before model loading.
        probe_key = f"{self.namespace}/preflight/write_read.json"
        payload = json.dumps({"experiment_id": self.experiment_id, "model": self.model, "time": time.time()}).encode()
        self.checkpoints.store.put_bytes(payload, probe_key, overwrite=True)
        if self.checkpoints.store.get_bytes(probe_key) != payload:
            raise RuntimeError("artifact store write/read verification failed")
        result = {"status": "ok", "model": self.model, "checkpoint": mcfg["checkpoint"],
                  "configured_revision": mcfg.get("revision"), "config_hash": self.cfg.hash,
                  "transformers": {"installed": installed_transformers, "minimum": mcfg["minimum_transformers"],
                                   "compatible": Version(installed_transformers) >= Version(str(mcfg["minimum_transformers"]))},
                  "container_digest": os.environ.get("CROSS_PROBE_CONTAINER_DIGEST"),
                  "inputs": inputs, "openrouter": credit, "gpu": gpu, "artifact_write_read": "ok"}
        write_json(self.control / "static_preflight.json", result)
        return result

    def _command(self, stage: str) -> list[list[str]]:
        py = sys.executable
        base = ["--config", str(self.cfg.path), "--model", self.model]
        cloud = self.cfg["cloud"]
        if stage == "prepare":
            return []
        if stage in ("organic_generation", "inorganic_generation", "activation_replay"):
            return [[py, str(PROJECT_ROOT / "scripts/33_cloud_model_stage.py"), *base,
                     "--experiment-id", self.experiment_id, "--stage", stage]]
        if stage in ("organic_judging", "inorganic_judging"):
            return [[py, str(PROJECT_ROOT / "scripts/04_judge.py"), *base, "--dataset", dataset,
                     "--tag", self.experiment_id, "--no-gate"] for dataset in cloud["datasets"]]
        if stage == "eligibility":
            return [[py, str(PROJECT_ROOT / "scripts/02b_select_known.py"), *base, "--dataset", dataset,
                     "--tag", self.experiment_id, "--criterion", "correct_never_wrong", "--no-gate"]
                    for dataset in cloud["datasets"]]
        if stage == "dataset_build":
            commands = []
            for dataset in cloud["datasets"]:
                commands.append([py, str(PROJECT_ROOT / "scripts/31_cloud_replay.py"), *base, "--dataset", dataset,
                                 "--tag", self.experiment_id, "--materialize-only", "--no-gate"])
                commands.append([py, str(PROJECT_ROOT / "scripts/05_build_dataset.py"), *base, "--dataset", dataset,
                                 "--tag", self.experiment_id, "--no-gate"])
            return commands
        if stage == "probe_fit":
            commands = []
            # Compatibility arrays are intentionally absent from GCS. Rebuild
            # them from immutable replay shards after a restore, or cheaply skip
            # when the local copies are already current.
            for dataset in cloud["datasets"]:
                commands.append([py, str(PROJECT_ROOT / "scripts/31_cloud_replay.py"), *base, "--dataset", dataset,
                                 "--tag", self.experiment_id, "--materialize-only", "--no-gate"])
                commands.append([py, str(PROJECT_ROOT / "scripts/05_build_dataset.py"), *base, "--dataset", dataset,
                                 "--tag", self.experiment_id, "--no-gate"])
            commands.append([py, str(PROJECT_ROOT / "scripts/32_cloud_probes.py"), *base,
                             "--experiment-id", self.experiment_id, "--phase", "fit", "--no-gate"])
            return commands
        if stage == "report":
            return [[py, str(PROJECT_ROOT / "scripts/32_cloud_probes.py"), *base,
                     "--experiment-id", self.experiment_id, "--phase", "report", "--no-gate"]]
        raise ValueError(stage)

    def _latest_mtime(self) -> float:
        latest = 0.0
        for root in self.local_roots.values():
            if root.exists():
                for path in root.rglob("*"):
                    if path.is_file():
                        latest = max(latest, path.stat().st_mtime)
        return latest

    def run_stage(self, stage: str, force: bool = False) -> None:
        if stage not in STAGES:
            raise ValueError(stage)
        if self.is_complete(stage) and not force:
            log.info("stage already complete: %s", stage)
            return
        monitor = RunMonitor(self.control / "monitoring" / stage, self.experiment_id, self.model, stage, self.cfg["cloud"])
        self.monitor = monitor
        monitor.start(int(self.cfg["cloud"]["heartbeat_seconds"]))
        self.checkpoints.start_periodic(stage, int(self.cfg["cloud"]["checkpoint_interval_seconds"]),
                                        lambda: monitor.progress_data)
        current: subprocess.Popen | None = None

        def terminate(signum, _frame):
            if current and current.poll() is None:
                current.terminate()
            try:
                self.checkpoints.checkpoint(stage, {**monitor.progress_data, "signal": signum})
            finally:
                raise SystemExit(128 + signum)

        old_term, old_int = signal.signal(signal.SIGTERM, terminate), signal.signal(signal.SIGINT, terminate)
        try:
            write_json(self.marker(stage), {"status": "running", "stage": stage, "config_hash": self.cfg.hash,
                                            "started_unix": time.time()})
            last_change, last_mtime = time.time(), self._latest_mtime()
            checkpoint_stale_alerted = False
            checkpoint_failure_alerted = False
            for number, command in enumerate(self._command(stage), start=1):
                monitor.progress(command_number=number, command_total=len(self._command(stage)), command=" ".join(command))
                current = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, bufsize=1)
                hardware_alerted: set[str] = set()

                def relay_output():
                    assert current is not None and current.stdout is not None
                    for line in current.stdout:
                        print(line, end="", flush=True)
                        lowered = line.casefold()
                        if ("cuda out of memory" in lowered or "outofmemoryerror" in lowered) and "oom" not in hardware_alerted:
                            hardware_alerted.add("oom")
                            monitor.alert("Cross-probe GPU out of memory", line.strip()[-1000:])
                        if "xid" in lowered and "nvidia" in lowered and "xid" not in hardware_alerted:
                            hardware_alerted.add("xid")
                            monitor.alert("Cross-probe NVIDIA Xid error", line.strip()[-1000:])

                relay = threading.Thread(target=relay_output, name="stage-output", daemon=True)
                relay.start()
                while current.poll() is None:
                    time.sleep(10)
                    mtime = self._latest_mtime()
                    if mtime > last_mtime:
                        last_mtime, last_change = mtime, time.time()
                    idle = time.time() - last_change
                    system = monitor.system_metrics()
                    checkpoint_age = (time.time() - self.checkpoints.last_success
                                      if self.checkpoints.last_success is not None else time.time() - read_json(self.marker(stage))["started_unix"])
                    monitor.progress(idle_seconds=idle, checkpoint_age_seconds=checkpoint_age,
                                     checkpoint_failures=self.checkpoints.consecutive_failures, **system)
                    if idle > int(self.cfg["cloud"]["stall_seconds"]):
                        monitor.alert("Cross-probe stage appears stalled",
                                      f"{stage} has produced no filesystem progress for {idle / 60:.1f} minutes")
                        last_change = time.time()  # alert once per stall interval
                    if checkpoint_age > int(self.cfg["cloud"]["checkpoint_max_age_seconds"]):
                        if not checkpoint_stale_alerted:
                            monitor.alert("Cross-probe checkpoint is stale",
                                          f"No verified checkpoint for {checkpoint_age / 60:.1f} minutes")
                            checkpoint_stale_alerted = True
                    else:
                        checkpoint_stale_alerted = False
                    if self.checkpoints.consecutive_failures >= 3 and not checkpoint_failure_alerted:
                        monitor.alert("Cross-probe checkpoints failing",
                                      f"{self.checkpoints.consecutive_failures} consecutive checkpoint failures")
                        checkpoint_failure_alerted = True
                    elif self.checkpoints.consecutive_failures == 0:
                        checkpoint_failure_alerted = False
                relay.join(timeout=30)
                if current.returncode:
                    raise subprocess.CalledProcessError(current.returncode, command)
            marker = {"status": "complete", "stage": stage, "config_hash": self.cfg.hash,
                      "completed_unix": time.time()}
            write_json(self.marker(stage), marker)
            if stage == "report":
                monitor.log_report(self.results)
            self.checkpoints.stop_periodic()
            self.checkpoints.checkpoint(stage, marker)
            monitor.stop("finished")
        except BaseException as exc:
            self.checkpoints.stop_periodic()
            try:
                self.checkpoints.checkpoint(stage, {**monitor.progress_data, "failed": True, "error": str(exc)})
            except Exception:
                pass
            monitor.alert("Cross-probe stage failed", f"{stage}: {type(exc).__name__}: {exc}")
            monitor.stop("failed", error=str(exc))
            raise
        finally:
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)
            self.monitor = None

    def run(self, start: str | None = None, stop: str | None = None) -> None:
        if not any(self.marker(stage).exists() for stage in STAGES):
            self.restore_latest(required=False)
        begin = STAGES.index(start) if start else 0
        end = STAGES.index(stop) + 1 if stop else len(STAGES)
        for stage in STAGES[begin:end]:
            self.run_stage(stage)

    def restore(self, stage: str) -> dict:
        return self.checkpoints.restore(stage)

    def restore_latest(self, required: bool = True) -> dict | None:
        for stage in reversed(STAGES):
            try:
                if self.checkpoints.latest(stage) is not None:
                    restored = self.checkpoints.restore(stage)
                    log.info("restored latest checkpoint from stage %s", stage)
                    return restored
            except FileNotFoundError:
                continue
        if required:
            raise FileNotFoundError(f"no checkpoints exist for {self.namespace}")
        return None

    def status(self) -> dict[str, Any]:
        return {stage: (read_json(self.marker(stage)) if self.marker(stage).exists() else {"status": "pending"})
                for stage in STAGES}

    def monitoring_preflight(self) -> None:
        """Exercise W&B, heartbeat publication, and both alert channels."""
        monitor = RunMonitor(self.control / "monitoring" / "preflight", self.experiment_id,
                             self.model, "preflight", self.cfg["cloud"])
        if self.cfg["cloud"].get("wandb", {}).get("enabled") and monitor.run is None:
            raise RuntimeError("W&B preflight failed to create a run")
        monitor.start(int(self.cfg["cloud"]["heartbeat_seconds"]))
        monitor.progress(preflight_complete=1)
        monitor.alert("Cross-probe preflight test",
                      f"Monitoring and alert delivery test for {self.experiment_id}/{self.model}", level="INFO")
        monitor.stop("finished")

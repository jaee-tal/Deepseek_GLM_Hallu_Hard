"""W&B tracking, portable heartbeats, and failure/stall alerts."""
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class RunMonitor:
    def __init__(self, root: Path, experiment_id: str, model: str, stage: str, config: dict[str, Any]):
        self.root, self.experiment_id, self.model, self.stage = root, experiment_id, model, stage
        self.config = config
        self.root.mkdir(parents=True, exist_ok=True)
        self.status_path = root / "status.json"
        self.last_progress = time.time()
        self.progress_data: dict[str, Any] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.run = None
        wcfg = config.get("wandb", {})
        wandb_mode = str(wcfg.get("mode", "online")).casefold()
        if wcfg.get("enabled") and (os.environ.get("WANDB_API_KEY") or wandb_mode == "offline"):
            try:
                import wandb
                self.run = wandb.init(project=wcfg.get("project", "cross-probe-cloud"),
                                      entity=wcfg.get("entity") or None,
                                      group=f"{experiment_id}-{model}",
                                      job_type=stage,
                                      id=f"{experiment_id}-{model}-{stage}".replace("/", "-"),
                                      resume="allow", mode=wcfg.get("mode", "online"),
                                      config={"experiment_id": experiment_id, "model": model, "stage": stage})
            except Exception:
                log.exception("W&B initialization failed")

    def _write_status(self, state: str, **extra) -> None:
        payload = {"experiment_id": self.experiment_id, "model": self.model, "stage": self.stage,
                   "state": state, "heartbeat_unix": time.time(), "pid": os.getpid(),
                   "progress": self.progress_data, **extra}
        tmp = self.status_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, self.status_path)
        print(json.dumps({"severity": "INFO", "event": "heartbeat", **payload}, default=str), flush=True)
        self._publish_gcp(payload)

    def _publish_gcp(self, payload: dict) -> None:
        project = self.config.get("monitoring", {}).get("gcp_project")
        if not project:
            return
        try:
            from google.cloud import monitoring_v3
            client = monitoring_v3.MetricServiceClient()
            prefix = self.config.get("monitoring", {}).get("metric_prefix", "custom.googleapis.com/cross_probe")
            labels = {"experiment_id": self.experiment_id, "model": self.model, "stage": self.stage}
            values = {"heartbeat": 1.0,
                      "idle_seconds": float(self.progress_data.get("idle_seconds", 0)),
                      "checkpoint_age_seconds": float(self.progress_data.get("checkpoint_age_seconds", 0)),
                      "disk_used_fraction": float(self.progress_data.get("system/disk_used_fraction", 0))}
            series_list = []
            for metric_name, value in values.items():
                series = monitoring_v3.TimeSeries()
                series.metric.type = f"{prefix}/{metric_name}"
                series.metric.labels.update(labels)
                series.resource.type = "global"
                series.resource.labels["project_id"] = project
                series.points = [monitoring_v3.Point({
                    "interval": {"end_time": {"seconds": int(time.time())}},
                    "value": {"double_value": value}})]
                series_list.append(series)
            client.create_time_series(name=f"projects/{project}", time_series=series_list)
        except Exception as exc:  # monitoring must not kill scientific work
            log.warning("Cloud Monitoring heartbeat failed: %s", exc)

    def start(self, seconds: int = 60) -> None:
        self._write_status("running")

        def loop():
            while not self._stop.wait(seconds):
                self._write_status("running")

        self._thread = threading.Thread(target=loop, name="heartbeat", daemon=True)
        self._thread.start()

    def progress(self, **values) -> None:
        self.progress_data.update(values)
        self.last_progress = time.time()
        if self.run:
            self.run.log(values)

    def checkpoint(self, manifest: dict) -> None:
        self.progress(checkpoint_age_seconds=0,
                      checkpoint_duration_seconds=manifest.get("duration_seconds"),
                      checkpoint_uploaded_bytes=manifest.get("uploaded_bytes"),
                      checkpoint_files=manifest.get("n_files"))

    def system_metrics(self) -> dict[str, float]:
        total, used, free = shutil.disk_usage(self.root)
        values = {"system/disk_used_fraction": used / total, "system/disk_free_gb": free / 1e9}
        try:
            import psutil
            values.update({"system/cpu_percent": psutil.cpu_percent(),
                           "system/ram_used_fraction": psutil.virtual_memory().percent / 100})
        except ImportError:
            pass
        return values

    def alert(self, title: str, text: str, level: str = "ERROR") -> None:
        print(json.dumps({"severity": level, "event": "alert", "title": title, "text": text,
                          "experiment_id": self.experiment_id, "model": self.model, "stage": self.stage}), flush=True)
        if self.run:
            try:
                import wandb
                self.run.alert(title=title, text=text, level=getattr(wandb.AlertLevel, level, wandb.AlertLevel.ERROR), wait_duration=300)
            except Exception:
                log.exception("W&B alert failed")

    def log_report(self, results_dir: Path) -> None:
        if not self.run:
            return
        try:
            import pandas as pd
            import wandb
            summary = results_dir / "equal_dataset_summary.csv"
            if summary.exists():
                frame = pd.read_csv(summary)
                self.run.log({"final/equal_dataset_summary": wandb.Table(dataframe=frame)})
                for _, row in frame.loc[frame.distribution == "organic_test"].iterrows():
                    prefix = f"final/{row['probe']}/{row['register']}/{row['arm']}"
                    self.run.summary[f"{prefix}/auroc"] = float(row["auroc"])
                    self.run.summary[f"{prefix}/precision"] = float(row["precision"])
                    self.run.summary[f"{prefix}/recall"] = float(row["recall"])
                    self.run.summary[f"{prefix}/balanced_accuracy"] = float(row["balanced_accuracy"])
            counts = results_dir / "sample_counts_by_dataset_and_label.csv"
            if counts.exists() and counts.stat().st_size:
                self.run.log({"final/sample_counts": wandb.Table(dataframe=pd.read_csv(counts))})
            images = {f"figures/{path.relative_to(results_dir).with_suffix('').as_posix()}": wandb.Image(str(path))
                      for path in results_dir.rglob("*.png")}
            if images:
                self.run.log(images)
        except Exception:
            log.exception("failed to upload final report to W&B")

    def stop(self, state: str = "finished", **extra) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        self._write_status(state, **extra)
        if self.run:
            self.run.finish(exit_code=0 if state == "finished" else 1)

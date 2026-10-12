"""Content-addressed, atomic checkpoints for mutable experiment workspaces."""
from __future__ import annotations

import fnmatch
import json
import logging
import os
import random
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .artifacts import ArtifactStore, file_sha256

log = logging.getLogger(__name__)

DEFAULT_EXCLUDES = (
    "*.tmp", "*.tmp.*", "*.lock", "logs/*", "**/logs/*",
    # These arrays are reproducible from immutable replay shards and would make
    # every hourly checkpoint copy tens of GB.
    "**/replay/activations.npy", "**/dataset/acts_layer_major.npy",
    "**/transfer_experiments/cache/acts_layer_major.npy",
)


def _rng_state() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": list(random.getstate()),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), numpy_state[2], numpy_state[3], numpy_state[4]],
        "torch_cpu": torch.get_rng_state().tolist(),
        "generation_note": "subject generation uses stable plan-batch seeds; cloud model batch_size is one",
    }


def _tuples(value):
    return tuple(_tuples(x) for x in value) if isinstance(value, list) else value


def _restore_rng_state(state: dict) -> None:
    random.setstate(_tuples(state["python"]))
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), int(n[2]), int(n[3]), float(n[4])))
    torch.set_rng_state(torch.tensor(state["torch_cpu"], dtype=torch.uint8))


@dataclass
class CheckpointManager:
    store: ArtifactStore
    namespace: str
    local_roots: dict[str, Path]
    config_hash: str
    model_revision: str | None = None
    container_digest: str | None = None
    excludes: tuple[str, ...] = DEFAULT_EXCLUDES
    on_checkpoint: Callable[[dict], None] | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _file_cache: dict[str, tuple[int, int, str, str]] = field(default_factory=dict, init=False)
    last_success: float | None = field(default=None, init=False)
    consecutive_failures: int = field(default=0, init=False)

    def _prefix(self, stage: str) -> str:
        return f"{self.namespace.strip('/')}/{stage}"

    def _excluded(self, logical: str) -> bool:
        return any(fnmatch.fnmatch(logical, pattern) for pattern in self.excludes)

    def _files(self):
        for label, root in sorted(self.local_roots.items()):
            if not root.exists():
                continue
            for path in sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink()):
                logical = f"{label}/{path.relative_to(root).as_posix()}"
                if not self._excluded(logical):
                    yield logical, path

    def checkpoint(self, stage: str, progress: dict | None = None) -> dict:
        with self._lock:
            started = time.time()
            entries = []
            uploaded = 0
            try:
                for logical, path in self._files():
                    before = path.stat()
                    cached = self._file_cache.get(logical)
                    if cached and cached[:2] == (before.st_size, before.st_mtime_ns):
                        entries.append({"path": logical, "sha256": cached[2], "size": before.st_size,
                                        "mtime_ns": before.st_mtime_ns, "object": cached[3]})
                        continue
                    digest = file_sha256(path)
                    after = path.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        # Append-only JSONL files can be active for hours. Capture
                        # the exact prefix visible at the initial stat rather than
                        # repeatedly skipping the experiment's most useful state.
                        if path.suffix != ".jsonl":
                            log.warning("skipping file changed during checkpoint: %s", path)
                            continue
                        with tempfile.NamedTemporaryFile(prefix="cross-probe-checkpoint-", delete=False) as snapshot:
                            snapshot_path = Path(snapshot.name)
                            remaining = before.st_size
                            with path.open("rb") as source:
                                while remaining:
                                    block = source.read(min(8 << 20, remaining))
                                    if not block:
                                        break
                                    snapshot.write(block)
                                    remaining -= len(block)
                        if remaining:
                            snapshot_path.unlink(missing_ok=True)
                            log.warning("skipping file truncated during checkpoint: %s", path)
                            continue
                        payload = snapshot_path.read_bytes()
                        boundary = payload.rfind(b"\n")
                        if boundary < 0:
                            snapshot_path.unlink(missing_ok=True)
                            continue
                        snapshot_path.write_bytes(payload[:boundary + 1])
                        upload_path = snapshot_path
                        digest = file_sha256(upload_path)
                        size = upload_path.stat().st_size
                    else:
                        upload_path = path
                        snapshot_path = None
                        size = after.st_size
                    key = f"objects/{digest[:2]}/{digest}"
                    existed = self.store.exists(key)
                    self.store.put_file(upload_path, key)
                    uploaded += 0 if existed else size
                    entries.append({"path": logical, "sha256": digest, "size": size,
                                    "mtime_ns": after.st_mtime_ns, "object": key})
                    if upload_path == path:
                        self._file_cache[logical] = (size, after.st_mtime_ns, digest, key)
                    if snapshot_path is not None:
                        snapshot_path.unlink(missing_ok=True)
                now = time.time()
                manifest = {
                    "version": 1, "namespace": self.namespace, "stage": stage,
                    "created_unix": now, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
                    "config_hash": self.config_hash, "model_revision": self.model_revision,
                    "container_digest": self.container_digest, "pid": os.getpid(),
                    "progress": progress or {}, "rng_state": _rng_state(), "files": entries,
                    "n_files": len(entries), "uploaded_bytes": uploaded,
                }
                payload = json.dumps(manifest, sort_keys=True, indent=2).encode()
                manifest_key = f"{self._prefix(stage)}/manifests/{int(now * 1000)}.json"
                self.store.put_bytes(payload, manifest_key)
                latest = json.dumps({"manifest": manifest_key, "sha256": __import__("hashlib").sha256(payload).hexdigest()},
                                    sort_keys=True).encode()
                self.store.put_bytes(latest, f"{self._prefix(stage)}/LATEST.json", overwrite=True)
                self.last_success = now
                self.consecutive_failures = 0
                manifest["duration_seconds"] = time.time() - started
                if self.on_checkpoint:
                    self.on_checkpoint(manifest)
                log.info("checkpoint %s: %d files, %.2f GiB new", stage, len(entries), uploaded / 2**30)
                return manifest
            except Exception:
                self.consecutive_failures += 1
                log.exception("checkpoint failed (%d consecutive)", self.consecutive_failures)
                raise

    def latest(self, stage: str) -> dict | None:
        key = f"{self._prefix(stage)}/LATEST.json"
        if not self.store.exists(key):
            return None
        pointer = json.loads(self.store.get_bytes(key))
        payload = self.store.get_bytes(pointer["manifest"])
        digest = __import__("hashlib").sha256(payload).hexdigest()
        if digest != pointer["sha256"]:
            raise RuntimeError("LATEST manifest checksum mismatch")
        return json.loads(payload)

    def restore(self, stage: str, *, verify_existing: bool = True) -> dict:
        manifest = self.latest(stage)
        if manifest is None:
            raise FileNotFoundError(f"no checkpoint exists for {self.namespace}/{stage}")
        for entry in manifest["files"]:
            label, relative = entry["path"].split("/", 1)
            if label not in self.local_roots:
                raise ValueError(f"manifest contains unknown local root {label!r}")
            target = self.local_roots[label] / relative
            if target.exists() and (not verify_existing or file_sha256(target) == entry["sha256"]):
                continue
            self.store.get_file(entry["object"], target)
            if file_sha256(target) != entry["sha256"]:
                target.unlink(missing_ok=True)
                raise RuntimeError(f"restored object failed SHA-256: {entry['path']}")
        if manifest.get("rng_state"):
            _restore_rng_state(manifest["rng_state"])
        return manifest

    def start_periodic(self, stage: str, interval_seconds: int, progress: Callable[[], dict]) -> None:
        if self._thread and self._thread.is_alive():
            raise RuntimeError("checkpoint worker already running")
        self._stop.clear()

        def loop():
            while not self._stop.wait(interval_seconds):
                try:
                    self.checkpoint(stage, progress())
                except Exception:  # already logged; watchdog handles repeated failures
                    pass

        self._thread = threading.Thread(target=loop, name="checkpoint-worker", daemon=True)
        self._thread.start()

    def stop_periodic(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None

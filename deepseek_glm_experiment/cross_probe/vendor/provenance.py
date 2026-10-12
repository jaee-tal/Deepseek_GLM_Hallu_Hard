"""Environment / model / benchmark provenance written into every run directory."""
from __future__ import annotations

import hashlib
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import PROJECT_ROOT, read_json, sha256_file, write_json


def _ver(mod: str) -> str | None:
    try:
        m = __import__(mod)
        return getattr(m, "__version__", None)
    except Exception:
        return None


def _git(cwd: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def collect(cfg_hash: str, cfg_path: str, model_cfg: dict[str, Any] | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config_path": str(cfg_path),
        "config_hash": cfg_hash,
        "versions": {m: _ver(m) for m in ("torch", "transformers", "accelerate", "huggingface_hub", "sklearn", "numpy", "pandas")},
    }
    try:
        import torch

        info["cuda"] = {
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        }
        drv = Path("/proc/driver/nvidia/version")
        if drv.exists():
            info["cuda"]["driver"] = drv.read_text().splitlines()[0]
    except Exception as e:  # pragma: no cover
        info["cuda"] = {"error": repr(e)}
    vend = Path(__file__).resolve().parent
    info["vendor"] = {"source": "bundled hallu_probe snapshot (2026-09-09)",
                      "files_sha256": {f.name: sha256_file(f) for f in sorted(vend.glob("*.py")) if f.name != "__init__.py"},
                      "source_sha256_list": (vend / "SOURCE_SHA256.txt").read_text() if (vend / "SOURCE_SHA256.txt").exists() else None}
    if model_cfg is not None:
        info["model"] = {k: v for k, v in model_cfg.items()}
    if extra:
        info.update(extra)
    return info


def write(run_dir: Path, info: dict[str, Any], section: str | None = None) -> None:
    """Merge `info` into run_dir/provenance.json (under `section` if given)."""
    p = run_dir / "provenance.json"
    cur = read_json(p) if p.exists() else {}
    if section:
        cur.setdefault(section, {}).update(info)
    else:
        cur.update(info)
    write_json(p, cur)

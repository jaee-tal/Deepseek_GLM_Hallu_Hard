"""Wait for enough free GPU memory before generation or activation replay."""
from __future__ import annotations

import logging
import subprocess
import sys
import time
from typing import Any

log = logging.getLogger(__name__)


def free_gpu_gb() -> float:
    """Query in a subprocess so this process holds no CUDA context while polling."""
    code = "import torch;f,t=torch.cuda.mem_get_info();print(f/1e9)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-400:])
    return float(result.stdout.strip())


def wait_for_gate(gate: dict[str, Any], require_gpu: bool = True) -> dict[str, Any]:
    if not require_gpu:
        return {"ok": True, "free_gpu_gb": None}
    required = float(gate["min_free_gpu_gb"])
    poll = float(gate.get("poll_seconds", 60))
    while True:
        free = free_gpu_gb()
        if free >= required:
            status = {"ok": True, "free_gpu_gb": free}
            log.info("GPU gate open: %.1f GB free", free)
            return status
        log.info("GPU gate waiting: %.1f GB free, %.1f GB required", free, required)
        time.sleep(poll)

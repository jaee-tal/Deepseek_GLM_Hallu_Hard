#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit(f"Python 3.11+ is required; found {sys.version.split()[0]}")
PY

mkdir -p data/hf_hub_cache data/hf_datasets_cache runs results cloud_artifacts wandb logs
if [[ -f BUNDLE_MANIFEST.sha256 ]]; then
  sha256sum --check --quiet BUNDLE_MANIFEST.sha256
  echo "Static bundle checksums passed."
fi
if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example; fill in credentials before an online run."
fi

.venv/bin/python scripts/verify_bundle.py

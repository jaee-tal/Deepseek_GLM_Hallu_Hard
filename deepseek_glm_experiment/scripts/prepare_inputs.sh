#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${CROSS_PROBE_PYTHON:-$ROOT/.venv/bin/python}"
for dataset in pop_qa trivia_qa hallu_hard; do
  "$PYTHON" scripts/01_prepare_questions.py --config configs/cloud_experiment.yaml \
    --model deepseek_v41_flash --dataset "$dataset" --tag fixed-inputs --no-gate
done
"$PYTHON" scripts/verify_bundle.py

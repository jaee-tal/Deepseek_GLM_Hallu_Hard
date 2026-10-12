#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${1:-}"
EXPERIMENT_ID="${2:-}"
if [[ "$MODEL" != "deepseek_v41_flash" && "$MODEL" != "glm_53_flash" ]]; then
  echo "Usage: $0 {deepseek_v41_flash|glm_53_flash} EXPERIMENT_ID [extra runner arguments]" >&2
  exit 2
fi
if [[ -z "$EXPERIMENT_ID" ]]; then
  echo "EXPERIMENT_ID is required." >&2
  exit 2
fi
shift 2

mkdir -p runs results cloud_artifacts wandb logs
PYTHON="${CROSS_PROBE_PYTHON:-$ROOT/.venv/bin/python}"
exec "$PYTHON" scripts/30_cloud_experiment.py run \
  --config configs/cloud_experiment.yaml \
  --experiment-id "$EXPERIMENT_ID" \
  --model "$MODEL" "$@"

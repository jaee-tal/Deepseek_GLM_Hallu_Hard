#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODEL="${1:-}"
EXPERIMENT_ID="${2:-}"
if [[ "$MODEL" != "deepseek_v41_flash" && "$MODEL" != "glm_53_flash" ]] || [[ -z "$EXPERIMENT_ID" ]]; then
  echo "Usage: $0 {deepseek_v41_flash|glm_53_flash} EXPERIMENT_ID" >&2
  exit 2
fi
exec "${CROSS_PROBE_PYTHON:-$ROOT/.venv/bin/python}" scripts/30_cloud_experiment.py status \
  --config configs/cloud_experiment.yaml --experiment-id "$EXPERIMENT_ID" --model "$MODEL"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_ID="${1:-}"
if [[ -z "$EXPERIMENT_ID" ]]; then
  echo "Usage: $0 EXPERIMENT_ID" >&2
  exit 2
fi
"$ROOT/scripts/run_model.sh" deepseek_v41_flash "$EXPERIMENT_ID"
"$ROOT/scripts/run_model.sh" glm_53_flash "$EXPERIMENT_ID"

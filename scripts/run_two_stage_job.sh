#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 CONFIG SUMMARY_JSON LOG_FILE" >&2
  exit 2
fi

CONFIG_PATH="$1"
SUMMARY_JSON="$2"
LOG_FILE="$3"

source /etc/profile.d/lmod.sh 2>/dev/null || true
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null || true

export MPLCONFIGDIR="/tmp/$(basename "${SUMMARY_JSON%.json}")"
mkdir -p "${MPLCONFIGDIR}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="src:${PYTHONPATH:-}"

/usr/bin/python3.11 -u scripts/run_two_stage_ground_state.py \
  --config "$CONFIG_PATH" \
  --stage-a-epochs 800 \
  --stage-a-min-ess 32 \
  --stage-a-min-energy 0.0 \
  --summary-json "$SUMMARY_JSON" | tee "$LOG_FILE"

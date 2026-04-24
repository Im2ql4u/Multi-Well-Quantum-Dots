#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 5 ]; then
  echo "usage: $0 STRATEGY CONFIG SUMMARY_JSON LOG_FILE GPU_ID" >&2
  exit 2
fi

STRATEGY="$1"
CONFIG_PATH="$2"
SUMMARY_JSON="$3"
LOG_FILE="$4"
GPU_ID="$5"

source /etc/profile.d/lmod.sh 2>/dev/null || true
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null || true

export CUDA_MANUAL_DEVICE="$GPU_ID"
export MPLCONFIGDIR="/tmp/$(basename "${SUMMARY_JSON%.json}")"
mkdir -p "${MPLCONFIGDIR}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="src:${PYTHONPATH:-}"

/usr/bin/python3.11 -u scripts/run_two_stage_research.py \
  --strategy "$STRATEGY" \
  --config "$CONFIG_PATH" \
  --stage-a-epochs 800 \
  --bootstrap-epochs 200 \
  --multistart-epochs 200 \
  --multistart-restarts 4 \
  --stage-a-min-ess 32 \
  --stage-a-min-energy 0.0 \
  --summary-json "$SUMMARY_JSON" | tee "$LOG_FILE"

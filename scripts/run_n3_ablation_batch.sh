#!/usr/bin/env bash
set -euo pipefail

ROOT="/itf-fi-ml/home/aleksns/Multi-Well-Quantum-Dots"
cd "$ROOT"

LOG="results/p4_n3_ablation_20260411.log"
: > "$LOG"

configs=(
  "configs/one_per_well/n3_ablation_pinn_reinforce_s42.yaml"
  "configs/one_per_well/n3_ablation_ctnn_reinforce_s42.yaml"
  "configs/one_per_well/n3_ablation_pinn_weakform_s42.yaml"
  "configs/one_per_well/n3_ablation_ctnn_weakform_s42.yaml"
)

for cfg in "${configs[@]}"; do
  echo "=== RUN $cfg ===" | tee -a "$LOG"
  if [[ ! -f "$cfg" ]]; then
    echo "Missing config: $cfg" | tee -a "$LOG"
    exit 2
  fi
  PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python src/run_ground_state.py --config "$cfg" 2>&1 | tee -a "$LOG"
done

echo "EXIT_CODE:0" | tee -a "$LOG"

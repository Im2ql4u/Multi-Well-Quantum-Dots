#!/usr/bin/env bash
set -euo pipefail

ROOT="/itf-fi-ml/home/aleksns/Multi-Well-Quantum-Dots"
cd "$ROOT"

LOG="results/p4_n3_setup_sweep_20260411.log"
: > "$LOG"

configs=(
  "configs/one_per_well/n3_setup_pinn_reinforce_mh20_dec2_coll512_s42.yaml"
  "configs/one_per_well/n3_setup_pinn_reinforce_mh40_dec3_coll512_s42.yaml"
  "configs/one_per_well/n3_setup_pinn_reinforce_mh20_dec2_coll512_lr5e4_s42.yaml"
)

for cfg in "${configs[@]}"; do
  echo "=== RUN $cfg ===" | tee -a "$LOG"
  [[ -f "$cfg" ]] || { echo "Missing config: $cfg" | tee -a "$LOG"; exit 2; }
  PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python src/run_ground_state.py --config "$cfg" 2>&1 | tee -a "$LOG"
done

echo "EXIT_CODE:0" | tee -a "$LOG"

#!/usr/bin/env bash
# Wait for N=2 target-J run (PID 837659) to finish, then launch N=4 d-sweep on GPU 3.
set -euo pipefail
cd /itf-fi-ml/home/aleksns/Multi-Well-Quantum-Dots

WAIT_PID=${WAIT_PID:-837659}
LOG=results/d_sweep/n4_uniform_s42_console.log
mkdir -p results/d_sweep

if [[ -n "${WAIT_PID}" && "${WAIT_PID}" != "0" ]]; then
  echo "[$(date -Is)] waiting on PID ${WAIT_PID} to finish before launching d-sweep..." > "${LOG}"
  while ps -p "${WAIT_PID}" >/dev/null 2>&1; do sleep 30; done
  echo "[$(date -Is)] PID ${WAIT_PID} finished, launching d-sweep" >> "${LOG}"
else
  echo "[$(date -Is)] no wait-pid set, launching d-sweep immediately" >> "${LOG}"
fi

CUDA_MANUAL_DEVICE=3 PYTHONPATH=src nohup python3.11 scripts/n_chain_d_sweep.py \
    --config configs/one_per_well/n4_invdes_baseline_s42.yaml \
    --n-wells 4 \
    --d-values 2.5 3.0 4.0 5.0 6.0 \
    --stage-a-epochs 2000 \
    --stage-b-epochs 1 \
    --stage-a-strategy improved_self_residual \
    --stage-a-min-energy 999.0 \
    --seed-override 42 \
    --device cuda:3 \
    --skip-existing \
    --out-dir results/d_sweep/n4_uniform_s42 \
    --log-level INFO >> "${LOG}" 2>&1

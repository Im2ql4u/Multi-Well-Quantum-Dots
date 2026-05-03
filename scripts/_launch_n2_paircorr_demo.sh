#!/usr/bin/env bash
# Phase 1C bilevel demo: drive g_sigma(r0=2.5) towards 0.40 by reshaping
# the N=2 dot via the d-parametrisation. Queued behind the N=4 d-sweep
# (and behind the N=2 target-J run that the d-sweep itself is queued
# behind) on GPU 3.
set -euo pipefail
cd /itf-fi-ml/home/aleksns/Multi-Well-Quantum-Dots

WAIT_PID=${WAIT_PID:-848666}
LOG=results/inverse_design/n2_paircorr_demo_s42_console.log
mkdir -p results/inverse_design

if [[ -n "${WAIT_PID}" && "${WAIT_PID}" != "0" ]]; then
  echo "[$(date -Is)] waiting on PID ${WAIT_PID} (parent launcher) before pair-corr demo..." > "${LOG}"
  while ps -p "${WAIT_PID}" >/dev/null 2>&1; do sleep 30; done
  echo "[$(date -Is)] PID ${WAIT_PID} finished, launching pair-corr demo" >> "${LOG}"
else
  echo "[$(date -Is)] no wait-pid set, launching pair-corr demo immediately" >> "${LOG}"
fi

CUDA_MANUAL_DEVICE=3 PYTHONPATH=src nohup python3.11 scripts/run_inverse_design.py \
    --config configs/one_per_well/n2_invdes_paircorr_baseline_s42.yaml \
    --target pair_corr \
    --parametrisation n2 \
    --param-init 4.0 --param-step 0.2 \
    --param-lower 1.5 --param-upper 6.0 \
    --r0 2.5 --sigma 0.30 \
    --mode neg_squared_error --target-value 0.40 \
    --n-corr-samples 4096 --corr-mh-warmup 400 --corr-mh-decorrelation 4 \
    --corr-seed 42 \
    --n-steps 5 --lr 4.0 --gradient-method fd_central \
    --stage-a-epochs 2000 --stage-b-epochs 1 \
    --stage-a-min-energy 999.0 \
    --stage-a-strategy improved_self_residual \
    --out-dir results/inverse_design/n2_paircorr_demo_s42 \
    --log-level INFO >> "${LOG}" 2>&1

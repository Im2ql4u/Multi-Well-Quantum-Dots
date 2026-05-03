#!/usr/bin/env bash
# Phase 2C MBL stretch demo: free 2D displacements per well, bilevel
# inverse design to maximise the bipartite well-set entanglement on
# N=4. With 8 free parameters and fd_forward, 9 inner trainings per
# outer step — about 30 min/step on N=4 (2000 epochs).
#
# Queued behind the pair-corr demo (which is queued behind the d-sweep,
# which is queued behind the N=2 target-J run) on GPU 3.
set -euo pipefail
cd /itf-fi-ml/home/aleksns/Multi-Well-Quantum-Dots

WAIT_PID=${WAIT_PID:-854332}
LOG=results/inverse_design/n4_displacement_demo_s42_console.log
mkdir -p results/inverse_design

if [[ -n "${WAIT_PID}" && "${WAIT_PID}" != "0" ]]; then
  echo "[$(date -Is)] waiting on PID ${WAIT_PID} (parent launcher) before displacement demo..." > "${LOG}"
  while ps -p "${WAIT_PID}" >/dev/null 2>&1; do sleep 30; done
  echo "[$(date -Is)] PID ${WAIT_PID} finished, launching displacement demo" >> "${LOG}"
else
  echo "[$(date -Is)] no wait-pid set, launching displacement demo immediately" >> "${LOG}"
fi

# 8 free params (4 wells x 2D), constrained to |delta| <= 0.6 ell_HO.
# Base layout = config wells (uniform N=4 chain at d=4: x=-6,-2,+2,+6).
CUDA_MANUAL_DEVICE=3 PYTHONPATH=src nohup python3.11 scripts/run_inverse_design.py \
    --config configs/one_per_well/n4_invdes_baseline_s42.yaml \
    --target well_set_entanglement \
    --metric von_neumann_entropy \
    --set-a 0 1 \
    --parametrisation displacement_2d \
    --param-step 0.10 0.10 0.10 0.10 0.10 0.10 0.10 0.10 \
    --param-lower -0.6 -0.6 -0.6 -0.6 -0.6 -0.6 -0.6 -0.6 \
    --param-upper +0.6 +0.6 +0.6 +0.6 +0.6 +0.6 +0.6 +0.6 \
    --n-steps 4 --lr 1.5 --gradient-method fd_forward \
    --stage-a-epochs 2000 --stage-b-epochs 1 \
    --stage-a-min-energy 999.0 \
    --stage-a-strategy improved_self_residual \
    --out-dir results/inverse_design/n4_displacement_demo_s42 \
    --log-level INFO >> "${LOG}" 2>&1

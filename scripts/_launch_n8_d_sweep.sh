#!/usr/bin/env bash
# Phase 3a — N=8 uniform-chain d-sweep on cuda:3.
# Test whether the surprising N=4 result (overlap with Heisenberg DECREASES
# as d grows beyond ~3 Bohr, indicating PINN variational bias) generalises
# to longer chains.
set -euo pipefail
cd /itf-fi-ml/home/aleksns/Multi-Well-Quantum-Dots

LOG=results/d_sweep/n8_uniform_s42_console.log
mkdir -p results/d_sweep

# We use the lean fast config (n_coll=384, hidden=96) that worked for the N=8
# SSH flagship. Stage-A epochs 1500 matches the flagship inner-loop budget.
CUDA_MANUAL_DEVICE=3 PYTHONPATH=src nohup python3.11 scripts/n_chain_d_sweep.py \
    --config configs/one_per_well/n8_invdes_fast_s42.yaml \
    --n-wells 8 \
    --d-values 2.5 3.0 4.0 5.0 6.0 \
    --stage-a-epochs 1500 \
    --stage-a-strategy improved_self_residual \
    --stage-a-min-energy 999.0 \
    --out-dir results/d_sweep/n8_uniform_s42 \
    > "${LOG}" 2>&1

#!/usr/bin/env bash
# Adiabatic connection lambda sweep — no E_ref, singlet_self_residual strategy.
# Trains N=2 singlet permanent at 5 Coulomb strengths (lambda=0.00–1.00)
# across 3 seeds for the correlation energy decomposition:
#   E_corr(lambda) = E(lambda) - E_HF(lambda)
# which is the key observable for Direction B (adiabatic connection).
#
# Strategy: singlet_self_residual — pure variance with singlet permanent + wider
# stratified sampler (no E_ref guidance at any lambda value).
#
# Lambda configs already have alpha_end=0.0 (no annealed CI target), so the
# stage-A strategy must be specified explicitly (auto would pick "guided").
#
# GPU assignments (15 jobs over 5 GPUs, 3 lambda values per GPU):
#   GPU0: lam=0.00 seeds 42,314,901
#   GPU1: lam=0.25 seeds 42,314,901
#   GPU2: lam=0.50 seeds 42,314,901
#   GPU3: lam=0.75 seeds 42,314,901
#   GPU4: lam=1.00 seeds 42,314,901
# Each set of 3 seeds runs sequentially on its GPU (one GPU per lambda).

set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
PYTHONPATH="$REPO/src"
RUNNER="$REPO/scripts/run_two_stage_ground_state.py"
CFG_DIR="$REPO/configs/magnetic"
STAGE_A_EPOCHS=2000
STAGE_B_EPOCHS=3000
STRATEGY="singlet_self_residual"
LOG_DIR="$REPO/logs/lambda_sweep_noref"
mkdir -p "$LOG_DIR"

echo "=== Launching no-ref lambda sweep at $(date) ==="

# Lambda configs: lam value → config file
declare -A LAM_CFGS=(
    ["0p00"]="$CFG_DIR/n2_singlet_d4_lam0p00_s42.yaml"
    ["0p25"]="$CFG_DIR/n2_singlet_d4_lam0p25_s42.yaml"
    ["0p50"]="$CFG_DIR/n2_singlet_d4_lam0p50_s42.yaml"
    ["0p75"]="$CFG_DIR/n2_singlet_d4_lam0p75_s42.yaml"
    ["1p00"]="$CFG_DIR/n2_singlet_d4_lam1p00_s42.yaml"
)

GPU=0
for LAM in 0p00 0p25 0p50 0p75 1p00; do
    CFG="${LAM_CFGS[$LAM]}"
    LOG_BASE="$LOG_DIR/lam${LAM}"
    nohup bash -c "
        for SEED in 42 314 901; do
            echo \"  lam=${LAM} seed=\$SEED GPU${GPU} starting at \$(date)\"
            CUDA_MANUAL_DEVICE=${GPU} PYTHONPATH=$PYTHONPATH python3.11 $RUNNER \\
                --config $CFG \\
                --stage-a-strategy $STRATEGY \\
                --stage-a-epochs $STAGE_A_EPOCHS \\
                --stage-b-epochs $STAGE_B_EPOCHS \\
                --seed-override \$SEED \\
                --stage-a-min-energy 0.5
            echo \"  lam=${LAM} seed=\$SEED done at \$(date)\"
        done
    " > "${LOG_BASE}.log" 2>&1 &
    PID=$!
    echo "lam=${LAM} (seeds 42,314,901) → GPU${GPU}, PID ${PID}"
    GPU=$((GPU + 1))
done

echo ""
echo "All 5 lambda groups launched (15 runs total)."
echo "Monitor with:"
echo "  tail -f $LOG_DIR/lam0p00.log"
echo "  tail -f $LOG_DIR/lam1p00.log"

#!/usr/bin/env bash
# No-ref N=2 singlet separation sweep.
# Tests how entanglement (and correlation energy) varies with well separation
# d = 2, 4, 6, 8, 12, 20 using the singlet_self_residual strategy (no CI reference).
#
# Key physics: at large d the system is a Mott singlet (antiferromagnet),
# entanglement S = log(2) ≈ 0.693; at small d the singlet permanent mixes
# with delocalized doubly-occupied configurations, S drops.
#
# GPU assignments (6 separations × 3 seeds sequential per GPU):
#   GPU0: d=2   GPU1: d=4   GPU2: d=6
#   GPU3: d=8   GPU4: d=12  GPU5: d=20

set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
PYTHONPATH="$REPO/src"
RUNNER="$REPO/scripts/run_two_stage_ground_state.py"
CFG_DIR="$REPO/configs/one_per_well"
STAGE_A_EPOCHS=2000
STAGE_B_EPOCHS=3000
STRATEGY="singlet_self_residual"
LOG_DIR="$REPO/logs/singlet_sep_sweep"
mkdir -p "$LOG_DIR"

echo "=== Launching singlet separation sweep at $(date) ==="

declare -A SEP_CFGS=(
    ["d2"]="$CFG_DIR/n2_singlet_d2_s42.yaml"
    ["d4"]="$CFG_DIR/n2_singlet_d4_s42.yaml"
    ["d6"]="$CFG_DIR/n2_singlet_d6_s42.yaml"
    ["d8"]="$CFG_DIR/n2_singlet_d8_s42.yaml"
    ["d12"]="$CFG_DIR/n2_singlet_d12_s42.yaml"
    ["d20"]="$CFG_DIR/n2_singlet_d20_s42.yaml"
)
declare -A SEP_GPU=(
    ["d2"]="0"
    ["d4"]="1"
    ["d6"]="2"
    ["d8"]="3"
    ["d12"]="4"
    ["d20"]="5"
)

for SEP in d2 d4 d6 d8 d12 d20; do
    CFG="${SEP_CFGS[$SEP]}"
    GPU="${SEP_GPU[$SEP]}"
    LOG_BASE="$LOG_DIR/${SEP}"
    nohup bash -c "
        for SEED in 42 314 901; do
            echo \"  sep=${SEP} seed=\$SEED GPU${GPU} starting at \$(date)\"
            CUDA_MANUAL_DEVICE=${GPU} PYTHONPATH=$PYTHONPATH python3.11 $RUNNER \\
                --config $CFG \\
                --stage-a-strategy $STRATEGY \\
                --stage-a-epochs $STAGE_A_EPOCHS \\
                --stage-b-epochs $STAGE_B_EPOCHS \\
                --seed-override \$SEED \\
                --stage-a-min-energy 0.5
            echo \"  sep=${SEP} seed=\$SEED done at \$(date)\"
        done
    " > "${LOG_BASE}.log" 2>&1 &
    PID=$!
    echo "sep=${SEP} (seeds 42,314,901) → GPU${GPU}, PID ${PID}"
done

echo ""
echo "All 6 separation groups launched (18 runs total)."
echo "Monitor with:"
echo "  tail -f $LOG_DIR/d4.log"
echo "  tail -f $LOG_DIR/d20.log"

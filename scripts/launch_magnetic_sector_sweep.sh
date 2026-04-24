#!/usr/bin/env bash
# Direction D: Magnetic phase diagram — N=3 and N=4 spin-sector sweep at B=0.5.
#
# Runs all 7 fixed-spin-sector configs with 3 seeds each using the no-ref
# improved_self_residual strategy. Comparing energies across sectors identifies
# the ground-state spin at B=0.5 (d=4, ω=1) without CI reference guidance.
#
# N=3 sectors: Sz = -3/2, -1/2, +1/2, +3/2  (0up, 1up, 2up, 3up)
# N=4 sectors: Sz = -2, 0, +2               (0up4down, 2up2down, 4up0down)
#
# Each Zeeman-shifted sector energy E(Sz) = E_corr - B * g * mu_B * Sz / 2
# so comparing E across sectors at fixed B gives the magnetic ground state.
#
# GPU assignments (7 sectors × 3 seeds sequential per GPU):
#   GPU0: N=3 0up3down   GPU1: N=3 1up2down   GPU2: N=3 2up1down
#   GPU3: N=3 3up0down   GPU4: N=4 0up4down
#   GPU5: N=4 2up2down   GPU6: N=4 4up0down

set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
PYTHONPATH="$REPO/src"
RUNNER="$REPO/scripts/run_two_stage_ground_state.py"
CFG_DIR="$REPO/configs/magnetic"
STAGE_A_EPOCHS=4000
STAGE_B_EPOCHS=3000
STRATEGY="improved_self_residual"
LOG_DIR="$REPO/logs/magnetic_sector_sweep"
mkdir -p "$LOG_DIR"

echo "=== Launching magnetic spin-sector sweep at $(date) ==="

# ---- N=3 sectors ----
declare -A N3_CFGS=(
    ["0up"]="$CFG_DIR/n3_0up3down_b0p5_s42.yaml"
    ["1up"]="$CFG_DIR/n3_1up2down_b0p5_s42.yaml"
    ["2up"]="$CFG_DIR/n3_2up1down_b0p5_s42.yaml"
    ["3up"]="$CFG_DIR/n3_3up0down_b0p5_s42.yaml"
)
declare -A N3_MIN_E=(
    ["0up"]="1.0"
    ["1up"]="2.0"
    ["2up"]="3.0"
    ["3up"]="4.0"
)
declare -A N3_GPU=(
    ["0up"]="0"
    ["1up"]="1"
    ["2up"]="2"
    ["3up"]="3"
)

for SECTOR in 0up 1up 2up 3up; do
    CFG="${N3_CFGS[$SECTOR]}"
    GPU="${N3_GPU[$SECTOR]}"
    MIN_E="${N3_MIN_E[$SECTOR]}"
    LOG_BASE="$LOG_DIR/n3_${SECTOR}"
    nohup bash -c "
        for SEED in 42 314 901; do
            echo \"  N=3 ${SECTOR} seed=\$SEED GPU${GPU} starting at \$(date)\"
            CUDA_MANUAL_DEVICE=${GPU} PYTHONPATH=$PYTHONPATH python3.11 $RUNNER \\
                --config $CFG \\
                --stage-a-strategy $STRATEGY \\
                --stage-a-epochs $STAGE_A_EPOCHS \\
                --stage-b-epochs $STAGE_B_EPOCHS \\
                --seed-override \$SEED \\
                --stage-a-min-energy $MIN_E
            echo \"  N=3 ${SECTOR} seed=\$SEED done at \$(date)\"
        done
    " > "${LOG_BASE}.log" 2>&1 &
    PID=$!
    echo "N=3 ${SECTOR} (seeds 42,314,901) → GPU${GPU}, PID ${PID}"
done

# ---- N=4 sectors ----
declare -A N4_CFGS=(
    ["0up4down"]="$CFG_DIR/n4_0up4down_b0p5_s42.yaml"
    ["2up2down"]="$CFG_DIR/n4_2up2down_b0p5_s42.yaml"
    ["4up0down"]="$CFG_DIR/n4_4up0down_b0p5_s42.yaml"
)
declare -A N4_MIN_E=(
    ["0up4down"]="2.0"
    ["2up2down"]="4.0"
    ["4up0down"]="5.5"
)
declare -A N4_GPU=(
    ["0up4down"]="4"
    ["2up2down"]="5"
    ["4up0down"]="6"
)

for SECTOR in 0up4down 2up2down 4up0down; do
    CFG="${N4_CFGS[$SECTOR]}"
    GPU="${N4_GPU[$SECTOR]}"
    MIN_E="${N4_MIN_E[$SECTOR]}"
    LOG_BASE="$LOG_DIR/n4_${SECTOR}"
    nohup bash -c "
        for SEED in 42 314 901; do
            echo \"  N=4 ${SECTOR} seed=\$SEED GPU${GPU} starting at \$(date)\"
            CUDA_MANUAL_DEVICE=${GPU} PYTHONPATH=$PYTHONPATH python3.11 $RUNNER \\
                --config $CFG \\
                --stage-a-strategy $STRATEGY \\
                --stage-a-epochs $STAGE_A_EPOCHS \\
                --stage-b-epochs $STAGE_B_EPOCHS \\
                --seed-override \$SEED \\
                --stage-a-min-energy $MIN_E
            echo \"  N=4 ${SECTOR} seed=\$SEED done at \$(date)\"
        done
    " > "${LOG_BASE}.log" 2>&1 &
    PID=$!
    echo "N=4 ${SECTOR} (seeds 42,314,901) → GPU${GPU}, PID ${PID}"
done

echo ""
echo "All 7 spin-sector groups launched (21 runs total)."
echo "Monitor with:"
echo "  tail -f $LOG_DIR/n3_0up.log"
echo "  tail -f $LOG_DIR/n3_3up.log"
echo "  tail -f $LOG_DIR/n4_2up2down.log"
echo "  tail -f $LOG_DIR/n4_4up0down.log"

#!/usr/bin/env bash
# Critical-field B-sweep for N=3 and N=4 magnetic phase transition.
# Sweeps B = 0.0, 0.003, 0.005, 0.01 to locate the AFM→FM transition.
# Estimated B_c ~ 0.005 Ha from the B=0.5 sector ladder.
#
# Strategy: improved_self_residual (no CI ref), 2 seeds per sector for speed.
# At B=0, ↑1↓2 and ↑2↓1 should be degenerate (SU(2) symmetry check).
# At B > B_c, ↑0↓N becomes GS; below B_c, the mixed-spin sector wins.
#
# N=3: 4 sectors × 4 B-values × 2 seeds = 32 runs → GPUs 0-7 (4 per GPU)
# Run N=3 first; launch N=4 separately once results are in.

set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
PYTHONPATH="$REPO/src"
RUNNER="$REPO/scripts/run_two_stage_ground_state.py"
CFG_DIR="$REPO/configs/magnetic"
STAGE_A_EPOCHS=3000
STAGE_B_EPOCHS=2000
STRATEGY="improved_self_residual"
LOG_DIR="$REPO/logs/critical_B_sweep"
mkdir -p "$LOG_DIR"

echo "=== Launching critical-B sweep at $(date) ==="

# N=3: 4 sectors × 4 B-values = 16 groups, each 2 seeds, assigned round-robin to 8 GPUs
# Group 2 B-values per GPU for the same sector to keep things organised
declare -a GROUPS=(
    "n3_0up3down_b0p0_s42.yaml  n3_0up3down_b0p003_s42.yaml  GPU=0  tag=n3_0up_b0-b003"
    "n3_0up3down_b0p005_s42.yaml n3_0up3down_b0p01_s42.yaml   GPU=1  tag=n3_0up_b005-b01"
    "n3_1up2down_b0p0_s42.yaml  n3_1up2down_b0p003_s42.yaml  GPU=2  tag=n3_1up_b0-b003"
    "n3_1up2down_b0p005_s42.yaml n3_1up2down_b0p01_s42.yaml   GPU=3  tag=n3_1up_b005-b01"
    "n3_2up1down_b0p0_s42.yaml  n3_2up1down_b0p003_s42.yaml  GPU=4  tag=n3_2up_b0-b003"
    "n3_2up1down_b0p005_s42.yaml n3_2up1down_b0p01_s42.yaml   GPU=5  tag=n3_2up_b005-b01"
    "n3_3up0down_b0p0_s42.yaml  n3_3up0down_b0p003_s42.yaml  GPU=6  tag=n3_3up_b0-b003"
    "n3_3up0down_b0p005_s42.yaml n3_3up0down_b0p01_s42.yaml   GPU=7  tag=n3_3up_b005-b01"
)

for GROUP in "${GROUPS[@]}"; do
    # Parse: cfg1 cfg2 GPU=N tag=T
    CFG1=$(echo "$GROUP" | awk '{print $1}')
    CFG2=$(echo "$GROUP" | awk '{print $2}')
    GPU=$(echo "$GROUP" | grep -oP 'GPU=\K[0-9]+')
    TAG=$(echo "$GROUP" | grep -oP 'tag=\K\S+')
    LOG="$LOG_DIR/${TAG}.log"

    nohup bash -c "
        for CFG in $CFG_DIR/$CFG1 $CFG_DIR/$CFG2; do
            for SEED in 42 314; do
                echo \"  \$(basename \$CFG) seed=\$SEED GPU${GPU} starting at \$(date)\"
                CUDA_MANUAL_DEVICE=${GPU} PYTHONPATH=$PYTHONPATH python3.11 $RUNNER \\
                    --config \$CFG \\
                    --stage-a-strategy $STRATEGY \\
                    --stage-a-epochs $STAGE_A_EPOCHS \\
                    --stage-b-epochs $STAGE_B_EPOCHS \\
                    --seed-override \$SEED \\
                    --stage-a-min-energy 0.5
                echo \"  \$(basename \$CFG) seed=\$SEED done at \$(date)\"
            done
        done
    " > "$LOG" 2>&1 &
    echo "group ${TAG} → GPU${GPU}, PID $!"
done

echo ""
echo "All 8 N=3 groups launched (32 runs total)."
echo "Monitor with: tail -f $LOG_DIR/n3_1up_b0-b003.log"

#!/usr/bin/env bash
# Direction E: 2D quantum dot arrays — N=4 on a 2×2 square plaquette.
# Two separations (d=4 and d=8) × 3 seeds, plus the 1D chain reference
# at equivalent spacing, to compare entanglement topology.
#
# Key comparison:
#   plaquette d=4: 4 nearest-neighbour bonds in a loop (frustrated geometry)
#   chain d=4:     3 nearest-neighbour bonds in a line (linear geometry)
#   → if entanglement depends on connectivity, plaquette S > chain S at same d
#
# All runs use improved_self_residual (no CI reference).
# 1D chain reference re-uses the existing n4 one-per-well config.
#
# GPU assignments (seeds 42,314,901 sequential per GPU):
#   GPU0: plaquette d=4    GPU1: plaquette d=8
#   GPU2: 1D chain d=4 reference

set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
PYTHONPATH="$REPO/src"
RUNNER="$REPO/scripts/run_two_stage_ground_state.py"
PLAQ_DIR="$REPO/configs/plaquette"
ONE_PER_WELL="$REPO/configs/one_per_well"
STAGE_A_EPOCHS=4000
STAGE_B_EPOCHS=3000
STRATEGY="improved_self_residual"
LOG_DIR="$REPO/logs/plaquette_sweep"
mkdir -p "$LOG_DIR"

echo "=== Launching 2D plaquette sweep at $(date) ==="

# Plaquette d=4  (GPU0)
nohup bash -c "
    for SEED in 42 314 901; do
        echo \"  plaq_d4 seed=\$SEED GPU0 starting at \$(date)\"
        CUDA_MANUAL_DEVICE=0 PYTHONPATH=$PYTHONPATH python3.11 $RUNNER \\
            --config $PLAQ_DIR/n4_2x2_d4_2up2down_s42.yaml \\
            --stage-a-strategy $STRATEGY \\
            --stage-a-epochs $STAGE_A_EPOCHS \\
            --stage-b-epochs $STAGE_B_EPOCHS \\
            --seed-override \$SEED \\
            --stage-a-min-energy 3.0
        echo \"  plaq_d4 seed=\$SEED done at \$(date)\"
    done
" > "$LOG_DIR/plaq_d4.log" 2>&1 &
PID_D4=$!
echo "plaquette d=4 (seeds 42,314,901) → GPU0, PID $PID_D4"

# Plaquette d=8  (GPU1)
nohup bash -c "
    for SEED in 42 314 901; do
        echo \"  plaq_d8 seed=\$SEED GPU1 starting at \$(date)\"
        CUDA_MANUAL_DEVICE=1 PYTHONPATH=$PYTHONPATH python3.11 $RUNNER \\
            --config $PLAQ_DIR/n4_2x2_d8_2up2down_s42.yaml \\
            --stage-a-strategy $STRATEGY \\
            --stage-a-epochs $STAGE_A_EPOCHS \\
            --stage-b-epochs $STAGE_B_EPOCHS \\
            --seed-override \$SEED \\
            --stage-a-min-energy 2.0
        echo \"  plaq_d8 seed=\$SEED done at \$(date)\"
    done
" > "$LOG_DIR/plaq_d8.log" 2>&1 &
PID_D8=$!
echo "plaquette d=8 (seeds 42,314,901) → GPU1, PID $PID_D8"

# 1D chain reference (same N=4, d=4 one-per-well)  (GPU2)
nohup bash -c "
    for SEED in 42 314 901; do
        echo \"  chain_d4 seed=\$SEED GPU2 starting at \$(date)\"
        CUDA_MANUAL_DEVICE=2 PYTHONPATH=$PYTHONPATH python3.11 $RUNNER \\
            --config $ONE_PER_WELL/n4_nonmcmc_residual_anneal_s42.yaml \\
            --stage-a-strategy $STRATEGY \\
            --stage-a-epochs $STAGE_A_EPOCHS \\
            --stage-b-epochs $STAGE_B_EPOCHS \\
            --seed-override \$SEED \\
            --stage-a-min-energy 3.0
        echo \"  chain_d4 seed=\$SEED done at \$(date)\"
    done
" > "$LOG_DIR/chain_d4_ref.log" 2>&1 &
PID_CHAIN=$!
echo "1D chain d=4 ref (seeds 42,314,901) → GPU2, PID $PID_CHAIN"

echo ""
echo "All 3 geometry groups launched (9 runs total)."
echo "Monitor with:"
echo "  tail -f $LOG_DIR/plaq_d4.log"
echo "  tail -f $LOG_DIR/plaq_d8.log"
echo "  tail -f $LOG_DIR/chain_d4_ref.log"
echo ""
echo "PIDs: $PID_D4 $PID_D8 $PID_CHAIN"

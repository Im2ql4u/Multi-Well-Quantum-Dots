#!/usr/bin/env bash
# Launch multi-reference no-ref two-stage runs for N=3 and N=4 across 6 GPUs.
# Strategy: improved_self_residual with multi_ref=True (set by _apply_improved_noref_recipe).
# The multi-reference Slater determinant sums over all C(N_wells,n_up) well-to-spin
# assignments, preventing the localized symmetry-breaking minimum that caused all
# single-reference N=3/4 runs to land in product states (S≈0.002).
#
# Stage A: 4000 epochs pure variance minimisation (no E_ref)
# Stage B: 3000 epochs refinement, initialised from Stage A checkpoint
# Seeds: 42, 314, 901
#
# GPU assignments:
#   GPU0: N=3 seed=42
#   GPU1: N=3 seed=314
#   GPU2: N=3 seed=901
#   GPU3: N=4 seed=42
#   GPU4: N=4 seed=314
#   GPU5: N=4 seed=901

set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
PYTHONPATH="$REPO/src"
RUNNER="$REPO/scripts/run_two_stage_ground_state.py"
N3_CFG="$REPO/configs/one_per_well/n3_nonmcmc_residual_anneal_s42.yaml"
N4_CFG="$REPO/configs/one_per_well/n4_nonmcmc_residual_anneal_s42.yaml"
STAGE_A_EPOCHS=4000
STAGE_B_EPOCHS=3000
STRATEGY="improved_self_residual"
LOG_DIR="$REPO/logs/multiref_noref_n3n4"
mkdir -p "$LOG_DIR"

echo "=== Launching multi-reference no-ref N=3/N=4 campaign at $(date) ==="

# N=3 seed=42  (GPU0)
nohup bash -c "CUDA_MANUAL_DEVICE=0 PYTHONPATH=$PYTHONPATH python3.11 $RUNNER \
    --config $N3_CFG \
    --stage-a-strategy $STRATEGY \
    --stage-a-epochs $STAGE_A_EPOCHS \
    --stage-b-epochs $STAGE_B_EPOCHS \
    --seed-override 42 \
    --stage-a-min-energy 1.0" \
    > "$LOG_DIR/n3_s42.log" 2>&1 &
PID_N3_S42=$!
echo "N=3 s42  → GPU0, PID $PID_N3_S42"

# N=3 seed=314  (GPU1)
nohup bash -c "CUDA_MANUAL_DEVICE=1 PYTHONPATH=$PYTHONPATH python3.11 $RUNNER \
    --config $N3_CFG \
    --stage-a-strategy $STRATEGY \
    --stage-a-epochs $STAGE_A_EPOCHS \
    --stage-b-epochs $STAGE_B_EPOCHS \
    --seed-override 314 \
    --stage-a-min-energy 1.0" \
    > "$LOG_DIR/n3_s314.log" 2>&1 &
PID_N3_S314=$!
echo "N=3 s314 → GPU1, PID $PID_N3_S314"

# N=3 seed=901  (GPU2)
nohup bash -c "CUDA_MANUAL_DEVICE=2 PYTHONPATH=$PYTHONPATH python3.11 $RUNNER \
    --config $N3_CFG \
    --stage-a-strategy $STRATEGY \
    --stage-a-epochs $STAGE_A_EPOCHS \
    --stage-b-epochs $STAGE_B_EPOCHS \
    --seed-override 901 \
    --stage-a-min-energy 1.0" \
    > "$LOG_DIR/n3_s901.log" 2>&1 &
PID_N3_S901=$!
echo "N=3 s901 → GPU2, PID $PID_N3_S901"

# N=4 seed=42  (GPU3)
nohup bash -c "CUDA_MANUAL_DEVICE=3 PYTHONPATH=$PYTHONPATH python3.11 $RUNNER \
    --config $N4_CFG \
    --stage-a-strategy $STRATEGY \
    --stage-a-epochs $STAGE_A_EPOCHS \
    --stage-b-epochs $STAGE_B_EPOCHS \
    --seed-override 42 \
    --stage-a-min-energy 1.0" \
    > "$LOG_DIR/n4_s42.log" 2>&1 &
PID_N4_S42=$!
echo "N=4 s42  → GPU3, PID $PID_N4_S42"

# N=4 seed=314  (GPU4)
nohup bash -c "CUDA_MANUAL_DEVICE=4 PYTHONPATH=$PYTHONPATH python3.11 $RUNNER \
    --config $N4_CFG \
    --stage-a-strategy $STRATEGY \
    --stage-a-epochs $STAGE_A_EPOCHS \
    --stage-b-epochs $STAGE_B_EPOCHS \
    --seed-override 314 \
    --stage-a-min-energy 1.0" \
    > "$LOG_DIR/n4_s314.log" 2>&1 &
PID_N4_S314=$!
echo "N=4 s314 → GPU4, PID $PID_N4_S314"

# N=4 seed=901  (GPU5)
nohup bash -c "CUDA_MANUAL_DEVICE=5 PYTHONPATH=$PYTHONPATH python3.11 $RUNNER \
    --config $N4_CFG \
    --stage-a-strategy $STRATEGY \
    --stage-a-epochs $STAGE_A_EPOCHS \
    --stage-b-epochs $STAGE_B_EPOCHS \
    --seed-override 901 \
    --stage-a-min-energy 1.0" \
    > "$LOG_DIR/n4_s901.log" 2>&1 &
PID_N4_S901=$!
echo "N=4 s901 → GPU5, PID $PID_N4_S901"

echo ""
echo "All 6 jobs launched. Monitor with:"
echo "  tail -f $LOG_DIR/n3_s42.log"
echo "  tail -f $LOG_DIR/n4_s42.log"
echo ""
echo "PIDs: $PID_N3_S42 $PID_N3_S314 $PID_N3_S901 $PID_N4_S42 $PID_N4_S314 $PID_N4_S901"

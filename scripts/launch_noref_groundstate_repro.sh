#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source /etc/profile.d/lmod.sh 2>/dev/null || true
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null || true

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="results/noref_repro_logs_${TS}"
mkdir -p "$LOG_DIR" results/diag_sweeps

echo "Launching no-ref two-stage ground-state reproduction at ${TS}"
echo "Logs: $LOG_DIR"

run_two_stage() {
  local gpu="$1"
  local config="$2"
  local tag="$3"
  local summary="results/diag_sweeps/${tag}__two_stage_summary_${TS}.json"
  local log="${LOG_DIR}/${tag}.log"
  {
    echo "[$(date +%H:%M:%S)] GPU${gpu} -> ${tag}"
    export CUDA_MANUAL_DEVICE="${gpu}"
    export PYTHONPATH="src:${PYTHONPATH:-}"
    export MPLCONFIGDIR="/tmp/mpl_noref_groundstate_${gpu}"
    mkdir -p "${MPLCONFIGDIR}"
    /usr/bin/python3.11 scripts/run_two_stage_ground_state.py \
      --config "${config}" \
      --stage-a-epochs 800 \
      --stage-a-min-ess 32 \
      --stage-a-min-energy 0.0 \
      --summary-json "${summary}"
    echo "[$(date +%H:%M:%S)] done ${tag}"
  } >"${log}" 2>&1
}

# Longest jobs on dedicated GPUs.
(
  run_two_stage 1 configs/one_per_well/n4_nonmcmc_residual_anneal_s42.yaml n4_seed42
) &

(
  run_two_stage 2 configs/one_per_well/seed_sweep/n4_nonmcmc_residual_anneal_s314.yaml n4_seed314
) &

(
  run_two_stage 4 configs/one_per_well/seed_sweep/n4_nonmcmc_residual_anneal_s901.yaml n4_seed901
) &

# Medium jobs, then short jobs on the same GPUs.
(
  run_two_stage 5 configs/one_per_well/n3_nonmcmc_residual_anneal_s42.yaml n3_seed42
  run_two_stage 5 configs/one_per_well/n2_nonmcmc_residual_anneal_s42.yaml n2_seed42
) &

(
  run_two_stage 7 configs/one_per_well/seed_sweep/n3_nonmcmc_residual_anneal_s314.yaml n3_seed314
  run_two_stage 7 configs/one_per_well/seed_sweep/n2_nonmcmc_residual_anneal_s314.yaml n2_seed314
) &

(
  run_two_stage 3 configs/one_per_well/seed_sweep/n3_nonmcmc_residual_anneal_s901.yaml n3_seed901
  run_two_stage 3 configs/one_per_well/seed_sweep/n2_nonmcmc_residual_anneal_s901.yaml n2_seed901
) &

wait

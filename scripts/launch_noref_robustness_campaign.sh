#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source /etc/profile.d/lmod.sh 2>/dev/null || true
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null || true

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="results/noref_robustness_logs_${TS}"
mkdir -p "$LOG_DIR" results/diag_sweeps results/imag_time_pinn

LOG="results/noref_robustness_campaign_${TS}.log"
echo "Launching no-ref robustness campaign at ${TS}" | tee "$LOG"
echo "Logs: ${LOG_DIR}" | tee -a "$LOG"

resolve_result_dir() {
  local summary_json="$1"
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); stage=d.get("stage_b") or d.get("stage_a"); print(stage["result_dir"])' "$summary_json"
}

run_n2_seed_pipeline() {
  local gpu="$1"
  local config="$2"
  local tag="$3"
  local summary="results/diag_sweeps/${tag}__two_stage_summary_${TS}.json"
  local pipeline_log="${LOG_DIR}/${tag}.log"
  local gs_comp_json="results/diag_sweeps/${tag}__components_${TS}.json"
  local gs_virial_json="results/diag_sweeps/${tag}__virial_${TS}.json"
  local gs_ent_n24="results/diag_sweeps/${tag}__gs_ent_n24_${TS}.json"
  local gs_ent_n28="results/diag_sweeps/${tag}__gs_ent_n28_${TS}.json"
  local quench_log="${LOG_DIR}/${tag}__quench.log"
  local quench_ent_tau0="results/diag_sweeps/${tag}__quench_tau0_ent_${TS}.json"
  local quench_ent_tau1="results/diag_sweeps/${tag}__quench_tau1_ent_${TS}.json"

  {
    echo "[$(date +%H:%M:%S)] GPU${gpu} -> ${tag} ground-state pipeline"
    export CUDA_MANUAL_DEVICE="${gpu}"
    export PYTHONPATH="src:${PYTHONPATH:-}"
    export MPLCONFIGDIR="/tmp/mpl_noref_robust_${gpu}"
    mkdir -p "${MPLCONFIGDIR}"

    python3 scripts/run_two_stage_ground_state.py \
      --config "${config}" \
      --stage-a-epochs 800 \
      --stage-a-min-ess 32 \
      --stage-a-min-energy 0.0 \
      --summary-json "${summary}"

    local result_dir
    result_dir="$(resolve_result_dir "${summary}")"
    echo "[$(date +%H:%M:%S)] ${tag} using result_dir=${result_dir}"

    python3 scripts/eval_ground_state_components.py \
      --result-dir "${result_dir}" \
      --device "cuda:${gpu}" \
      --n-samples 4096 \
      --mh-steps 20 \
      > "${gs_comp_json}"
    echo "[$(date +%H:%M:%S)] saved ${gs_comp_json}"

    python3 scripts/check_virial_multiwell.py \
      --result-dir "${result_dir}" \
      --device "cuda:${gpu}" \
      --output "${gs_virial_json}"
    echo "[$(date +%H:%M:%S)] saved ${gs_virial_json}"

    python3 scripts/measure_entanglement.py \
      --result-dir "${result_dir}" \
      --npts 24 \
      --device "cuda:${gpu}" \
      --batch-size 256 \
      --out "${gs_ent_n24}"
    python3 scripts/measure_entanglement.py \
      --result-dir "${result_dir}" \
      --npts 28 \
      --device "cuda:${gpu}" \
      --batch-size 256 \
      --out "${gs_ent_n28}"
    echo "[$(date +%H:%M:%S)] saved ${gs_ent_n24}"
    echo "[$(date +%H:%M:%S)] saved ${gs_ent_n28}"

    python3 src/imaginary_time_pinn.py \
      --quench_B 0.5 \
      --quench_profile fast \
      --ground_state_dir "${result_dir}" \
      --zeeman_electron1_only \
      > "${quench_log}" 2>&1
    echo "[$(date +%H:%M:%S)] quench log ${quench_log}"

    local quench_json
    local quench_ckpt
    quench_json="$(sed -n 's/^Saved single-B quench result: //p' "${quench_log}" | tail -n 1)"
    quench_ckpt="$(sed -n 's/^Checkpoint saved: *//p' "${quench_log}" | tail -n 1)"
    if [[ -z "${quench_json}" || -z "${quench_ckpt}" ]]; then
      echo "[$(date +%H:%M:%S)] ERROR failed to parse quench outputs for ${tag}" >&2
      exit 1
    fi
    echo "[$(date +%H:%M:%S)] ${tag} quench_json=${quench_json}"
    echo "[$(date +%H:%M:%S)] ${tag} quench_ckpt=${quench_ckpt}"

    python3 scripts/measure_entanglement.py \
      --quench-checkpoint "${quench_ckpt}" \
      --tau 0.0 \
      --npts 24 \
      --device "cuda:${gpu}" \
      --batch-size 256 \
      --out "${quench_ent_tau0}"
    python3 scripts/measure_entanglement.py \
      --quench-checkpoint "${quench_ckpt}" \
      --tau 1.0 \
      --npts 24 \
      --device "cuda:${gpu}" \
      --batch-size 256 \
      --out "${quench_ent_tau1}"
    echo "[$(date +%H:%M:%S)] saved ${quench_ent_tau0}"
    echo "[$(date +%H:%M:%S)] saved ${quench_ent_tau1}"

    echo "[$(date +%H:%M:%S)] done ${tag}"
  } > "${pipeline_log}" 2>&1
}

run_serial_ground_states() {
  local gpu="$1"
  local log_name="$2"
  shift 2
  local log_path="${LOG_DIR}/${log_name}.log"
  {
    export CUDA_MANUAL_DEVICE="${gpu}"
    export PYTHONPATH="src:${PYTHONPATH:-}"
    export MPLCONFIGDIR="/tmp/mpl_noref_robust_${gpu}"
    mkdir -p "${MPLCONFIGDIR}"
    while (($#)); do
      local cfg="$1"
      shift
      echo "[$(date +%H:%M:%S)] GPU${gpu} -> ${cfg}"
      python3 src/run_ground_state.py --config "${cfg}"
    done
    echo "[$(date +%H:%M:%S)] done ${log_name}"
  } > "${log_path}" 2>&1
}

# GPU 0/1/2: fresh N=2 no-ref seeds with downstream checks.
( run_n2_seed_pipeline 0 configs/one_per_well/n2_nonmcmc_residual_anneal_s42.yaml n2_seed42 ) &
echo "GPU0: N=2 seed42 pipeline pid=$!" | tee -a "$LOG"

( run_n2_seed_pipeline 1 configs/one_per_well/seed_sweep/n2_nonmcmc_residual_anneal_s314.yaml n2_seed314 ) &
echo "GPU1: N=2 seed314 pipeline pid=$!" | tee -a "$LOG"

( run_n2_seed_pipeline 2 configs/one_per_well/seed_sweep/n2_nonmcmc_residual_anneal_s901.yaml n2_seed901 ) &
echo "GPU2: N=2 seed901 pipeline pid=$!" | tee -a "$LOG"

# GPU 4/7: magnetic fixed-spin ladders from the canonical reproduction matrix.
( run_serial_ground_states 4 n3_magnetic \
  configs/magnetic/n3_3up0down_b0p5_s42.yaml \
  configs/magnetic/n3_2up1down_b0p5_s42.yaml \
  configs/magnetic/n3_1up2down_b0p5_s42.yaml \
  configs/magnetic/n3_0up3down_b0p5_s42.yaml ) &
echo "GPU4: N=3 magnetic ladder pid=$!" | tee -a "$LOG"

( run_serial_ground_states 7 n4_magnetic \
  configs/magnetic/n4_4up0down_b0p5_s42.yaml \
  configs/magnetic/n4_2up2down_b0p5_s42.yaml \
  configs/magnetic/n4_0up4down_b0p5_s42.yaml ) &
echo "GPU7: N=4 magnetic ladder pid=$!" | tee -a "$LOG"

# CPU sidecar: parameter-matched exact-diag magnetic reference for the N=2 quench protocol.
( export PYTHONPATH=src:${PYTHONPATH:-}; python3 scripts/run_magnetic_reference_sweep.py \
  --separations 4 \
  --B-pre 0.0 \
  --B-post-values 0.5 \
  --kappa 1.0 \
  --epsilon 0.01 \
  --output-prefix magnetic_reference_n2_matched_noref_campaign_${TS} \
  --summary-json results/diag_sweeps/magnetic_reference_n2_matched_noref_campaign_${TS}.json \
  > "${LOG_DIR}/magnetic_reference.log" 2>&1 ) &
echo "CPU: N=2 magnetic reference sweep pid=$!" | tee -a "$LOG"

echo "Campaign launched." | tee -a "$LOG"
echo "Monitor with: tail -f ${LOG_DIR}/*.log" | tee -a "$LOG"
wait

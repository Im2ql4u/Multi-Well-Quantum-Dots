#!/usr/bin/env bash
# 10-hour campaign launch script.
# GPU assignments (by free memory): GPU6=11002MB, GPU1/2/4/5/7=~10512MB, GPU3=9795MB, GPU0=8372MB
# Runs:
#   GPU6: N=8 s42 (8000ep, ~4hr)
#   GPU1: N=8 s314 (8000ep, ~4hr)
#   GPU2: N=8 s901 (8000ep, ~4hr)
#   GPU4: N=10 s42  (6000ep, ~5hr)
#   GPU5: N=10 s314 (6000ep, ~5hr)
#   GPU7: N=3 magnetic 4 spin sectors serial (4×4000ep, ~6hr total)
#   GPU3: N=4 magnetic 3 spin sectors serial (3×4000ep, ~4.5hr), then B quench
#   GPU0: N=2 singlet lambda-sweep (4×3000ep, ~2hr), then N=6+N=8 entanglement

set -euo pipefail
cd "$(dirname "$0")/.."
LOG=results/campaign_10hr_$(date +%Y%m%d_%H%M%S).log
mkdir -p results/diag_sweeps
echo "Campaign started: $(date)" | tee "$LOG"

# ── GPU 6: N=8 s42 ──────────────────────────────────────────────────────────
nohup bash -c "CUDA_MANUAL_DEVICE=6 PYTHONPATH=src python3.11 src/run_ground_state.py \
    --config configs/one_per_well/n8_nonmcmc_residual_s42.yaml" \
    > results/gpu6_n8_s42.log 2>&1 &
echo "GPU6: N=8 s42 pid=$!" | tee -a "$LOG"

# ── GPU 1: N=8 s314 ──────────────────────────────────────────────────────────
nohup bash -c "CUDA_MANUAL_DEVICE=1 PYTHONPATH=src python3.11 src/run_ground_state.py \
    --config configs/one_per_well/n8_nonmcmc_residual_s314.yaml" \
    > results/gpu1_n8_s314.log 2>&1 &
echo "GPU1: N=8 s314 pid=$!" | tee -a "$LOG"

# ── GPU 2: N=8 s901 ──────────────────────────────────────────────────────────
nohup bash -c "CUDA_MANUAL_DEVICE=2 PYTHONPATH=src python3.11 src/run_ground_state.py \
    --config configs/one_per_well/n8_nonmcmc_residual_s901.yaml" \
    > results/gpu2_n8_s901.log 2>&1 &
echo "GPU2: N=8 s901 pid=$!" | tee -a "$LOG"

# ── GPU 4: N=10 s42 ──────────────────────────────────────────────────────────
nohup bash -c "CUDA_MANUAL_DEVICE=4 PYTHONPATH=src python3.11 src/run_ground_state.py \
    --config configs/one_per_well/n10_nonmcmc_selfconsistent_s42.yaml" \
    > results/gpu4_n10_s42.log 2>&1 &
echo "GPU4: N=10 s42 pid=$!" | tee -a "$LOG"

# ── GPU 5: N=10 s314 ─────────────────────────────────────────────────────────
nohup bash -c "CUDA_MANUAL_DEVICE=5 PYTHONPATH=src python3.11 src/run_ground_state.py \
    --config configs/one_per_well/n10_nonmcmc_selfconsistent_s314.yaml" \
    > results/gpu5_n10_s314.log 2>&1 &
echo "GPU5: N=10 s314 pid=$!" | tee -a "$LOG"

# ── GPU 7: N=3 magnetic 4 sectors (serial) ───────────────────────────────────
nohup bash -c "
CUDA_MANUAL_DEVICE=7 PYTHONPATH=src python3.11 src/run_ground_state.py \
    --config configs/magnetic/n3_3up0down_b0p5_s42.yaml && \
CUDA_MANUAL_DEVICE=7 PYTHONPATH=src python3.11 src/run_ground_state.py \
    --config configs/magnetic/n3_2up1down_b0p5_s42.yaml && \
CUDA_MANUAL_DEVICE=7 PYTHONPATH=src python3.11 src/run_ground_state.py \
    --config configs/magnetic/n3_1up2down_b0p5_s42.yaml && \
CUDA_MANUAL_DEVICE=7 PYTHONPATH=src python3.11 src/run_ground_state.py \
    --config configs/magnetic/n3_0up3down_b0p5_s42.yaml
" > results/gpu7_n3_magnetic.log 2>&1 &
echo "GPU7: N=3 magnetic 4 sectors pid=$!" | tee -a "$LOG"

# ── GPU 3: N=4 magnetic 3 sectors (serial), then B=0->0.5 quench ─────────────
N4_ARTIFACT=results/p4_n4_nonmcmc_residual_anneal_s42_20260416_113343
nohup bash -c "
CUDA_MANUAL_DEVICE=3 PYTHONPATH=src python3.11 src/run_ground_state.py \
    --config configs/magnetic/n4_4up0down_b0p5_s42.yaml && \
CUDA_MANUAL_DEVICE=3 PYTHONPATH=src python3.11 src/run_ground_state.py \
    --config configs/magnetic/n4_2up2down_b0p5_s42.yaml && \
CUDA_MANUAL_DEVICE=3 PYTHONPATH=src python3.11 src/run_ground_state.py \
    --config configs/magnetic/n4_0up4down_b0p5_s42.yaml && \
CUDA_MANUAL_DEVICE=3 PYTHONPATH=src python3.11 src/imaginary_time_pinn.py \
    --quench_B 0.5 \
    --ground_state_dir $N4_ARTIFACT \
    --zeeman_particles 0,1 \
    --quench_profile fast
" > results/gpu3_n4_mag_quench.log 2>&1 &
echo "GPU3: N=4 magnetic+quench pid=$!" | tee -a "$LOG"

# ── GPU 0: N=2 lambda-sweep, then entanglement measurements ──────────────────
N6_ART=results/p5_n6_nonmcmc_residual_anneal_s42_20260419_144602
N6_ART_S314=results/p5_n6_nonmcmc_residual_anneal_s314_20260419_202704
N6_ART_S901=results/p5_n6_nonmcmc_residual_anneal_s901_20260420_010721
N8_OLD=results/p5_n8_nonmcmc_residual_s42_20260414_134404

nohup bash -c "
CUDA_MANUAL_DEVICE=0 PYTHONPATH=src python3.11 src/run_ground_state.py \
    --config configs/magnetic/n2_singlet_d4_lam0p00_s42.yaml && \
CUDA_MANUAL_DEVICE=0 PYTHONPATH=src python3.11 src/run_ground_state.py \
    --config configs/magnetic/n2_singlet_d4_lam0p25_s42.yaml && \
CUDA_MANUAL_DEVICE=0 PYTHONPATH=src python3.11 src/run_ground_state.py \
    --config configs/magnetic/n2_singlet_d4_lam0p50_s42.yaml && \
CUDA_MANUAL_DEVICE=0 PYTHONPATH=src python3.11 src/run_ground_state.py \
    --config configs/magnetic/n2_singlet_d4_lam0p75_s42.yaml && \
echo '--- N=6 entanglement bipartitions ---' && \
PYTHONPATH=src python3.11 scripts/measure_entanglement.py \
    --result-dir $N6_ART --npts 7 --device cuda:0 \
    --partition-particles '0,1,2' \
    --out results/diag_sweeps/n6_s42_ent_3v3_$(date +%Y%m%d).json && \
PYTHONPATH=src python3.11 scripts/measure_entanglement.py \
    --result-dir $N6_ART --npts 7 --device cuda:0 \
    --partition-particles '0' \
    --out results/diag_sweeps/n6_s42_ent_1v5_$(date +%Y%m%d).json && \
PYTHONPATH=src python3.11 scripts/measure_entanglement.py \
    --result-dir $N6_ART --npts 7 --device cuda:0 \
    --partition-particles '0,1' \
    --out results/diag_sweeps/n6_s42_ent_2v4_$(date +%Y%m%d).json && \
PYTHONPATH=src python3.11 scripts/measure_entanglement.py \
    --result-dir $N6_ART_S314 --npts 7 --device cuda:0 \
    --partition-particles '0,1,2' \
    --out results/diag_sweeps/n6_s314_ent_3v3_$(date +%Y%m%d).json && \
PYTHONPATH=src python3.11 scripts/measure_entanglement.py \
    --result-dir $N6_ART_S901 --npts 7 --device cuda:0 \
    --partition-particles '0,1,2' \
    --out results/diag_sweeps/n6_s901_ent_3v3_$(date +%Y%m%d).json && \
echo '--- N=8 entanglement from old artifact ---' && \
PYTHONPATH=src python3.11 scripts/measure_entanglement.py \
    --result-dir $N8_OLD --npts 5 --device cuda:0 \
    --partition-particles '0,1,2,3' \
    --out results/diag_sweeps/n8_old_ent_4v4_$(date +%Y%m%d).json && \
PYTHONPATH=src python3.11 scripts/measure_entanglement.py \
    --result-dir $N8_OLD --npts 5 --device cuda:0 \
    --partition-particles '0' \
    --out results/diag_sweeps/n8_old_ent_1v7_$(date +%Y%m%d).json && \
PYTHONPATH=src python3.11 scripts/measure_entanglement.py \
    --result-dir $N8_OLD --npts 5 --device cuda:0 \
    --partition-particles '0,1' \
    --out results/diag_sweeps/n8_old_ent_2v6_$(date +%Y%m%d).json
" > results/gpu0_lamsweep_ent.log 2>&1 &
echo "GPU0: lambda-sweep + entanglement pid=$!" | tee -a "$LOG"

echo "All processes launched. Monitor with:" | tee -a "$LOG"
echo "  tail -f results/gpu*.log" | tee -a "$LOG"
echo "  nvidia-smi" | tee -a "$LOG"

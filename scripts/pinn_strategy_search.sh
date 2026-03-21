#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="results/imag_time_runs/strategy_search_${STAMP}"
mkdir -p "$OUT_DIR"

LOG_FILE="$OUT_DIR/search.log"
SCORE_FILE="$OUT_DIR/strategy_scores.tsv"
TOP_FILE="$OUT_DIR/top_strategies.tsv"
RANK_FILE="$OUT_DIR/ranked_strategies.tsv"

cat > "$SCORE_FILE" <<'TSV'
combo	profile	coulomb	loss	tau_sampling	x_sampling	mae	run_dir
TSV

echo "# Strategy search started at $(date)" | tee -a "$LOG_FILE"
echo "# Output dir: $OUT_DIR" | tee -a "$LOG_FILE"

LOSSES=(curriculum_mse mse huber logcosh)
TAUS=(small_bias uniform two_stage)
XS=(uniform energy_weighted)

MAX_COMBOS="${MAX_COMBOS:-0}"
RUN_TIMEOUT="${RUN_TIMEOUT:-70m}"
STAGE1_PROFILE="${STAGE1_PROFILE:-tiny}"
DISTANCES_OFF="${DISTANCES_OFF:-0,2,4,8}"
DISTANCES_ON="${DISTANCES_ON:-2,4,8,12}"
DO_STAGE2="${DO_STAGE2:-1}"
STAGE2_PROFILE="${STAGE2_PROFILE:-baseline}"
TOP_K="${TOP_K:-4}"

run_and_score() {
  local profile="$1"
  local coulomb_flag="$2"
  local combo="$3"
  local loss="$4"
  local tau="$5"
  local x="$6"
  local distances="$7"
  local run_tag="sweep_${combo}_${profile}_${coulomb_flag}_"

  echo "[$(date +%F\ %T)] Running combo=$combo profile=$profile coulomb=$coulomb_flag loss=$loss tau=$tau x=$x" | tee -a "$LOG_FILE"

  local before after run_dir
  before="$(ls -1 results/imag_time_runs 2>/dev/null | wc -l)"

  if [[ "$coulomb_flag" == "true" ]]; then
    timeout "$RUN_TIMEOUT" python3.11 src/run_imaginary_time.py \
      --mode run \
      --profile "$profile" \
      --strategies pinn \
      --distances "$distances" \
      --coulomb \
      --tag "$run_tag" \
      --pinn-loss-style "$loss" \
      --pinn-tau-sampling "$tau" \
      --pinn-x-sampling "$x" \
      --pinn-no-vmc-train \
      >> "$LOG_FILE" 2>&1 || true
  else
    timeout "$RUN_TIMEOUT" python3.11 src/run_imaginary_time.py \
      --mode run \
      --profile "$profile" \
      --strategies pinn \
      --distances "$distances" \
      --no-coulomb \
      --tag "$run_tag" \
      --pinn-loss-style "$loss" \
      --pinn-tau-sampling "$tau" \
      --pinn-x-sampling "$x" \
      --pinn-no-vmc-train \
      >> "$LOG_FILE" 2>&1 || true
  fi

  after="$(ls -1 results/imag_time_runs 2>/dev/null | wc -l)"
  if [[ "$after" -le "$before" ]]; then
    echo "[$(date +%F\ %T)] No new run dir detected for combo=$combo" | tee -a "$LOG_FILE"
    return 0
  fi

  run_dir="$(ls -1dt results/imag_time_runs/*/ | head -n 1 | sed 's:/$::')"

  local summary_path=""
  if [[ -f "$run_dir/summary_table.json" ]]; then
    summary_path="$run_dir/summary_table.json"
  elif [[ -f "$run_dir/summary.json" ]]; then
    summary_path="$run_dir/summary.json"
  fi

  if [[ -z "$summary_path" ]]; then
    echo "[$(date +%F\ %T)] Missing summary output in $run_dir" | tee -a "$LOG_FILE"
    return 0
  fi

  local mae
  mae="$(python3.11 - "$summary_path" "$coulomb_flag" <<'PY'
import json
import math
import sys

path = sys.argv[1]
coulomb = sys.argv[2].lower() == "true"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Support both legacy dict summary.json and current list summary_table.json.
if isinstance(data, dict):
    per = data.get("per_distance", [])
elif isinstance(data, list):
    per = data
else:
    per = []

if not per:
    print("nan")
    sys.exit(0)

errs = []
for row in per:
    d = float(row.get("distance", row.get("d", 0.0)))
    e = row.get("best_energy_fit", row.get("E_vmc"))
    if e is None:
        continue
    e = float(e)
    if coulomb:
        ref = 3.0 if d <= 1e-12 else (2.0 + 1.0 / d)
    else:
        ref = 2.0
    errs.append(abs(e - ref))

if not errs:
    print("nan")
else:
    print(f"{sum(errs)/len(errs):.8f}")
PY
)"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$combo" "$profile" "$coulomb_flag" "$loss" "$tau" "$x" "$mae" "$run_dir" \
    >> "$SCORE_FILE"

  echo "[$(date +%F\ %T)] Scored combo=$combo mae=$mae run_dir=$run_dir" | tee -a "$LOG_FILE"
}

# Stage 1: broad screening.
combo_idx=0
for loss in "${LOSSES[@]}"; do
  for tau in "${TAUS[@]}"; do
    for x in "${XS[@]}"; do
      combo_idx=$((combo_idx + 1))
      if [[ "$MAX_COMBOS" -gt 0 && "$combo_idx" -gt "$MAX_COMBOS" ]]; then
        continue
      fi
      combo_id="c$(printf "%02d" "$combo_idx")"
      run_and_score "$STAGE1_PROFILE" false "$combo_id" "$loss" "$tau" "$x" "$DISTANCES_OFF"
      run_and_score "$STAGE1_PROFILE" true  "$combo_id" "$loss" "$tau" "$x" "$DISTANCES_ON"
    done
  done
done

# Rank stage-1 results by MAE and select top 4 combos.
python3.11 - "$SCORE_FILE" "$RANK_FILE" "$TOP_FILE" "$TOP_K" <<'PY'
import csv
import math
import sys
from collections import defaultdict

score_path, rank_path, top_path = sys.argv[1:4]

rows = []
with open(score_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for r in reader:
        try:
            mae = float(r["mae"])
        except Exception:
            continue
        if math.isnan(mae):
            continue
        r["mae"] = mae
        rows.append(r)

rows_sorted = sorted(rows, key=lambda r: r["mae"])
with open(rank_path, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["combo", "profile", "coulomb", "loss", "tau_sampling", "x_sampling", "mae", "run_dir"])
    for r in rows_sorted:
        w.writerow([r["combo"], r["profile"], r["coulomb"], r["loss"], r["tau_sampling"], r["x_sampling"], f"{r['mae']:.8f}", r["run_dir"]])

agg = defaultdict(list)
for r in rows:
    key = (r["combo"], r["loss"], r["tau_sampling"], r["x_sampling"])
    agg[key].append(r["mae"])

combo_rank = []
for key, vals in agg.items():
    combo_rank.append((sum(vals) / len(vals), key))
combo_rank.sort(key=lambda x: x[0])

with open(top_path, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["combo", "loss", "tau_sampling", "x_sampling", "avg_mae_stage1"])
    top_k = 4
    try:
        top_k = max(1, int(sys.argv[4]))
    except Exception:
        pass

    for avg, (combo, loss, tau, x) in combo_rank[:top_k]:
        w.writerow([combo, loss, tau, x, f"{avg:.8f}"])
PY

# Stage 2: refine top 4 with baseline profile if time remains.
if [[ "$DO_STAGE2" == "1" && -f "$TOP_FILE" ]]; then
  tail -n +2 "$TOP_FILE" | while IFS=$'\t' read -r combo loss tau x avg; do
    run_and_score "$STAGE2_PROFILE" false "${combo}_b" "$loss" "$tau" "$x" "$DISTANCES_OFF"
    run_and_score "$STAGE2_PROFILE" true  "${combo}_b" "$loss" "$tau" "$x" "$DISTANCES_ON"
  done
fi

echo "# Strategy search finished at $(date)" | tee -a "$LOG_FILE"
echo "# Ranked output: $RANK_FILE" | tee -a "$LOG_FILE"

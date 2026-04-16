#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_campaign.sh --config-dir <dir> --gpus <g0,g1,...> --session-prefix <prefix> --log-dir <dir>
  scripts/run_campaign.sh --status --session-prefix <prefix>

Options:
  --config-dir       Directory containing YAML configs to run
  --gpus             Comma-separated GPU IDs (example: 0,1,5,6)
  --session-prefix   tmux session prefix (example: wknd_p2)
  --log-dir          Directory for per-run logs
  --status           Show status for sessions with prefix
  --dry-run          Print launch commands without executing
  -h, --help         Show this message
EOF
}

CONFIG_DIR=""
GPUS=""
PREFIX=""
LOG_DIR=""
STATUS_ONLY=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-dir) CONFIG_DIR="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --session-prefix) PREFIX="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --status) STATUS_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 2 ;;
  esac
done

if [[ $STATUS_ONLY -eq 1 ]]; then
  if [[ -z "$PREFIX" ]]; then
    echo "--session-prefix is required with --status"
    exit 2
  fi
  tmux ls 2>/dev/null | grep "^${PREFIX}_" || echo "No sessions found for prefix ${PREFIX}_"
  exit 0
fi

if [[ -z "$CONFIG_DIR" || -z "$GPUS" || -z "$PREFIX" || -z "$LOG_DIR" ]]; then
  usage
  exit 2
fi

if [[ ! -d "$CONFIG_DIR" ]]; then
  echo "Config directory not found: $CONFIG_DIR"
  exit 2
fi

mkdir -p "$LOG_DIR"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
mapfile -t CFGS < <(find "$CONFIG_DIR" -maxdepth 1 -type f -name '*.yaml' | sort)

if [[ ${#CFGS[@]} -eq 0 ]]; then
  echo "No .yaml configs found in $CONFIG_DIR"
  exit 2
fi

for i in "${!CFGS[@]}"; do
  cfg="${CFGS[$i]}"
  gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
  base="$(basename "$cfg" .yaml)"
  session="${PREFIX}_${base}"
  log_file="${LOG_DIR}/${base}.log"
  cmd="cd /itf-fi-ml/home/aleksns/Multi-Well-Quantum-Dots && PYTHONPATH=src CUDA_VISIBLE_DEVICES=${gpu} .venv/bin/python -u src/run_ground_state.py --config ${cfg} | tee ${log_file}"

  echo "[$((i+1))/${#CFGS[@]}] gpu=${gpu} session=${session} cfg=${cfg}"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "  tmux new-session -d -s ${session} \"${cmd}\""
    continue
  fi

  tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -d -s "$session" "$cmd"
done

echo "Launched ${#CFGS[@]} runs with prefix ${PREFIX}_"

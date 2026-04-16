#!/usr/bin/env bash
set -euo pipefail

ROOT="/itf-fi-ml/home/aleksns/Multi-Well-Quantum-Dots"
cd "$ROOT"

GPUS="2,3,4,7"
LOGROOT="results/weekend_pipeline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGROOT"

wait_prefix_done() {
  local prefix="$1"
  while true; do
    if ! tmux ls 2>/dev/null | grep -q "^${prefix}_"; then
      break
    fi
    sleep 30
  done
}

launch_batch_dir() {
  local dir="$1"
  local prefix="$2"
  local logdir="$3"
  mkdir -p "$logdir"
  scripts/run_campaign.sh --config-dir "$dir" --gpus "$GPUS" --session-prefix "$prefix" --log-dir "$logdir"
}

make_batches() {
  local src_dir="$1"
  local out_root="$2"
  mkdir -p "$out_root/b1" "$out_root/b2" "$out_root/b3"
  mapfile -t files < <(find "$src_dir" -maxdepth 1 -type f -name '*.yaml' | sort)
  local i=0
  for f in "${files[@]}"; do
    local b=$(( i / 4 + 1 ))
    cp -f "$f" "$out_root/b${b}/"
    i=$((i+1))
  done
}

# Phase 1
launch_batch_dir "configs/weekend_exec/phase1" "wk2_p1" "$LOGROOT/phase1"
wait_prefix_done "wk2_p1"

python3 - <<'PY' > "$LOGROOT/phase1_summary.txt"
import glob, json, os
rows=[]
for p in sorted(glob.glob('results/wk2_p1_*/result.json')):
    d=json.load(open(p))
    rows.append((os.path.dirname(p), d.get('final_energy'), d.get('final_energy_var')))
for r in rows:
    print('%s | E=%.6f var=%.6f' % r)
PY

# Phase 2 in 3 batches of 4 runs
make_batches "configs/weekend_exec/phase2" "$LOGROOT/phase2_batches"
for b in b1 b2 b3; do
  launch_batch_dir "$LOGROOT/phase2_batches/$b" "wk2_p2_${b}" "$LOGROOT/phase2_${b}"
  wait_prefix_done "wk2_p2_${b}"
done

python3 - <<'PY'
import glob, json, math
best=None
for p in sorted(glob.glob('results/wk2_arch_*/result.json')):
    d=json.load(open(p))
    e=float(d.get('final_energy', float('nan')))
    v=float(d.get('final_energy_var', float('nan')))
    if not math.isfinite(e) or not math.isfinite(v):
        continue
    # infer arch from run_name dir path
    name=p.split('/')[-2]
    if '_pinn_' in name:
        arch='pinn'
    elif '_ctnn_' in name:
        arch='ctnn'
    elif '_unified_' in name:
        arch='unified'
    else:
        continue
    score=v
    if best is None or score < best[0]:
        best=(score,arch,name,e,v)
if best is None:
    arch='pinn'
else:
    arch=best[1]
open('results/.wk2_best_arch.txt','w').write(arch)
print('best_arch', arch)
PY

BEST_ARCH=$(cat results/.wk2_best_arch.txt)

# Phase 3 config generation (8 runs)
python3 - <<'PY'
import yaml
from pathlib import Path
arch=open('results/.wk2_best_arch.txt').read().strip()
out=Path('configs/weekend_exec/phase3')
out.mkdir(parents=True, exist_ok=True)
for N in [2,4]:
    for omega in [0.5,1.0]:
        for seed in [501,502]:
            name=f'wk2_p3_n{N}_w{str(omega).replace(".","p")}_{arch}_s{seed}'
            if N==2:
                sys=dict(type='double_dot', n_left=1, n_right=1, separation=4.0, omega=omega, dim=2, coulomb=True)
            else:
                sys=dict(type='double_dot', n_left=2, n_right=2, separation=4.0, omega=omega, dim=2, coulomb=True)
            cfg=dict(
                run_name=name,
                allow_missing_dmc=True,
                system=sys,
                architecture=dict(arch_type=arch, pinn_hidden=64, pinn_layers=2, bf_hidden=64, bf_layers=2, use_backflow=True),
                training=dict(epochs=50000, lr=1e-4, lr_warmup_epochs=1000, lr_min_factor=0.1, n_coll=256, n_cand_mult=8, loss_type='fd_colloc', fd_h=0.01, sampler='mh', mh_steps=10, mh_step_scale=0.15, mh_decorrelation=1, grad_clip=0.1, print_every=100, seed=seed, device='cuda:0', dtype='float64')
            )
            with open(out/f'{name}.yaml','w') as f:
                yaml.safe_dump(cfg,f,default_flow_style=False)
print('phase3_configs', len(list(out.glob('*.yaml'))))
PY

# Phase 3 in 2 batches of 4
mkdir -p "$LOGROOT/phase3_batches/b1" "$LOGROOT/phase3_batches/b2"
mapfile -t p3files < <(find configs/weekend_exec/phase3 -maxdepth 1 -type f -name '*.yaml' | sort)
for i in "${!p3files[@]}"; do
  if (( i < 4 )); then
    cp -f "${p3files[$i]}" "$LOGROOT/phase3_batches/b1/"
  else
    cp -f "${p3files[$i]}" "$LOGROOT/phase3_batches/b2/"
  fi
done
for b in b1 b2; do
  launch_batch_dir "$LOGROOT/phase3_batches/$b" "wk2_p3_${b}" "$LOGROOT/phase3_${b}"
  wait_prefix_done "wk2_p3_${b}"
done

echo "Weekend pipeline complete. Logs: $LOGROOT"

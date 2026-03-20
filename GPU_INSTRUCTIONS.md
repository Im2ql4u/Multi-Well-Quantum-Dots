# GPU Environment Instructions

## 1. Clone and enter repository
```bash
git clone <your-repo-url> thesis-double-well
cd thesis-double-well
```

## 2. Create Python environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

## 3. Install dependencies
If you already have a lockfile or requirements export, use it. Otherwise install from `pyproject.toml`:
```bash
pip install -e .
```

If `pip install -e .` fails due to missing project metadata tooling, fallback:
```bash
pip install torch numpy scipy matplotlib
```

## 4. Quick sanity checks
```bash
python -m py_compile src/run_imaginary_time.py src/imaginary_time_vmc.py src/imaginary_time_pinn.py
python src/run_imaginary_time.py --mode smoke --strategies pinn --distances 0 --no-coulomb --tag gpu_smoke_
```

## 5. Recommended run progression
1. Tiny validation:
```bash
python src/run_imaginary_time.py --mode run --profile tiny --strategies vmc,pinn --distances 0,4 --coulomb --tag gpu_tiny_
```

2. Baseline check:
```bash
python src/run_imaginary_time.py --mode run --profile baseline --strategies vmc,pinn --distances 0,4,8 --coulomb --tag gpu_base_
```

3. Production run:
```bash
python src/run_imaginary_time.py --mode run --profile production --strategies vmc,pinn --distances 0,4,8 --coulomb --tag gpu_prod_
```

## 6. Monitoring
```bash
# latest run directories
ls -1t results/imag_time_runs | head

# inspect one run
find results/imag_time_runs/<run_dir> -maxdepth 2 -type f | sort
```

## 7. Cleanup/archival helper
Archive previous imaginary-time outputs safely:
```bash
python src/run_imaginary_time.py --mode clean
```

## 8. Outputs to collect
For each completed run dir in `results/imag_time_runs/<timestamp>_<profile>/`:
- `suite_results.json`
- `summary_table.json`
- `run_manifest.json`
- generated `figure_*.png`

## 9. Notes
- Use `--coulomb` for interacting runs.
- Use `--no-coulomb` for free/reference checks.
- If running on multi-GPU, pin the desired device using your environment policy before launch.

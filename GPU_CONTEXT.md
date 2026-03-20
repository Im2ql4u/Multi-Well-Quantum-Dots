# GPU Migration Context

## Repository state
- This repo now includes a unified imaginary-time runner: `src/run_imaginary_time.py`.
- Legacy imaginary-time outputs were archived (not deleted) under `results/imag_time_archive/`.
- New orchestration outputs and smoke results are under `results/imag_time_runs/`.
- Production CPU run was intentionally stopped to move execution to a GPU environment.

## Key files
- `src/run_imaginary_time.py`: unified entry point for strategy runs and plotting.
- `src/imaginary_time_vmc.py`: tau-conditioned VMC imaginary-time TDSE pipeline.
- `src/imaginary_time_pinn.py`: spectral PINN imaginary-time TDSE pipeline.
- `results/imag_time_runs/imag_time_test_plan.md`: testing strategy checklist.
- `results/well_separation/per_d_results_v2.json`: latest double-well per-d production results.

## Ground-state per-d results snapshot
- d=0.0: E=3.00206
- d=1.0: E=2.48513
- d=2.0: E=2.33563
- d=4.0: E=2.24855
- d=6.0: E=2.16795
- d=8.0: E=2.12644
- d=12.0: E=2.08335
- d=16.0: E=2.06258
- d=20.0: E=2.04979

Interpretation: at large d, energy follows approximately 2 + 1/d due to Coulomb tail.

## Important run modes
`src/run_imaginary_time.py` supports:
- `--mode plan`: write/update test plan markdown.
- `--mode clean`: archive old imaginary-time result folders/files.
- `--mode smoke`: fast end-to-end validation run.
- `--mode run`: full configured run.

Profiles:
- `smoke`, `tiny`, `baseline`, `production`

Strategies:
- `vmc`, `pinn`

## Last CPU production command (stopped)
`/Users/aleksandersekkelsten/thesis-double-well/.venv/bin/python src/run_imaginary_time.py --mode run --profile production --strategies vmc,pinn --distances 0,4,8 --coulomb --tag prod_tdse_`

## Expected heavy runtime note
The production suite can be long on CPU. GPU execution is recommended for practical throughput.

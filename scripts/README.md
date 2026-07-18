# Script guide

The files directly under this directory reproduce the current trajectory-based
full Navier--Stokes study. Run every command from the repository root.

## Current pipeline

1. Data: `generate_data.py`, `prepare_cache.py`
2. Discovery: `discover_equation.py`, `discover_noise.py`
3. Training: `run_experiment.py`, `run_suite.py`
4. Extended experiments: `run_baselines.py`, `run_relation_ablation.py`,
   `run_noise.py`
5. Analysis and figures: `diagnose_predictability.py`, `plot_results.py`
6. Validation: `validate_science.py`, `validate_extensions.py`,
   `validate_paper.py`

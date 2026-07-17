# Script guide

The files directly under this directory reproduce the current trajectory-based
full Navier--Stokes study. Run every command from the repository root.

## Current pipeline

1. Data: `generate_hit_ns_v2.py`, `prepare_hit_ns_v2_cache.py`
2. Discovery: `discover_hit_ns_v2.py`, `discover_hit_ns_v3_noise.py`
3. Training: `run_full_ns_v2.py`, `run_full_ns_v2_suite.py`
4. Extended experiments: `run_full_ns_v3_baselines.py`,
   `run_full_ns_v3_equation_ablation.py`, `run_full_ns_v3_noise.py`
5. Analysis and figures: `diagnose_hit_ns_v2_prediction.py`,
   `plot_full_ns_v2.py`
6. Validation: `validate_full_ns_v2.py`,
   `validate_full_ns_v3_extensions.py`, `validate_paper.py`

`legacy/` preserves the earlier single-snapshot/spatial-relation pipeline. It
is retained for provenance but is not evidence for the current manuscript.

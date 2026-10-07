# Script guide

The files directly under this directory reproduce the current trajectory-based
full Navier--Stokes study. Run every command from the repository root.

## Current pipeline

1. Data: `generate_data.py`, `prepare_cache.py`
2. Discovery: `discover_equation.py`, `discover_noise.py`
3. Training: `run_experiment.py`, `run_suite.py`
4. Extended experiments: `run_baselines.py`, `run_lambda_ablation.py`,
   `run_relation_ablation.py`, `run_decoder_ablation.py`, `run_noise.py`
5. Analysis: `diagnose_predictability.py`, `analyze_latents.py`
6. Manuscript values and figures: `sync_paper_results.py`, `plot_results.py`
7. Scientific validation: `validate_science.py`
8. Local manuscript checks: `validate_extensions.py`, `validate_paper.py`

Training and extension scripts default to `--input-size 32 --target-size 16
--context-encoder residual_warmstart --lambda-phys 0.1`. Size-specific aggregate
filenames keep these results separate from historical 16³ artifacts. Existing
compatible runs are reused; missing runs are trained with seeds 42, 123 and 456.

Run `sync_paper_results.py` after all seven learning aggregates are complete.
It verifies protocol metadata and recomputes the seed means and sample standard
deviations before generating `paper/experiment_values.tex` and a hashed evidence
report. `--check` verifies those files without changing them. Scientific prose
remains in the manuscript and is reviewed against the measured results.

`plot_results.py --protocol v5` dispatches the matching scripts in
`paper/figure_code/`; `--protocol old16` explicitly selects historical data.

The Python plotting utilities are included in Git. Generated figures, English
and Chinese manuscript sources/PDFs, bibliography files, and reference audits
stay local before manuscript release. `validate_extensions.py` and
`validate_paper.py` require the author's local TeX source; they are additional
checks for that workflow.

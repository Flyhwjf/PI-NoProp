# Physics-Informed NoProp (PI-NoProp)

[GitHub repository](https://github.com/Flyhwjf/PI-NoProp)

This repository contains the HIT-based implementation, SPIDER equation
discovery pipelines, optimized local NoProp training code, experiment artifacts,
and the accompanying paper. The current path generates trajectory-disjoint
decaying HIT and discovers the complete time-dependent momentum equation before
transferring its measured coefficients into a first-frame physical condition
and strictly local NoProp objectives. The corrected noise-to-label schedule and
equation-conditioned 3-D encoder raise three-seed trajectory-disjoint accuracy
from 17.59% to 86.73% (low enstrophy) and from 17.46% to 82.94% (high
enstrophy). The v4 condition pretraining jointly optimizes the spatial encoder,
physical-rate encoder, and their fusion before strictly local block training.

## Directory guide

| Path | Purpose | Maintenance rule |
| --- | --- | --- |
| `src/` | Models, data loaders, physics losses, and SPIDER implementation | Source code |
| `scripts/` | Current full-NS data, discovery, training, plotting, and validation entry points | Source code |
| `tests/` | Optimized-pipeline unit tests | Source code |
| `data/generated_hit_ns/` | Dense, independent decaying-HIT trajectories for full-NS discovery | Regenerable large data |
| `data/cache_hit_ns/` | Trajectory-disjoint predictive-learning cache | Regenerable large data |
| `outputs/spider/` | Validated discovered-equation artifacts | Required by SPIDER-informed training |
| `outputs/runs/` | Per-run configurations and metrics | Experiment record |
| `outputs/models/` | Trained model weights | Large experiment artifact |
| `outputs/aggregate/` | Aggregated paper results | Experiment record |
| `paper/` | Current TeX source, figures, and compiled PDF | Current manuscript |
| `初始/` | Original supervisor-provided TeX and reference output | Read-only reference |

The current code assumes it is launched from the repository root. In particular,
the relative paths `data/cache_hit_ns`, `data/generated_hit_ns`, and `outputs` are part of the
runtime interface and should not be renamed casually.

## Validation

Use the project environment rather than the system Python:

```powershell
$python = 'E:\research\code\miniconda\Asaved\envs\maclearn\python.exe'
& $python scripts\validate_paper.py
& $python scripts\validate_science.py
& $python scripts\validate_extensions.py
& $python -m unittest discover -s tests -v
```

Compile the manuscript from `paper/`:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error "Physics-Informed NoProp.tex"
```

See `OPTIMIZATION.md` for the optimized training design and commands.

## Full Navier--Stokes pipeline

The solver uses correctly normalized 3/2 de-aliasing, RK4 integration and no
unobserved forcing.  Discovery, validation and test sets contain different DNS
trajectories.  The SPIDER library evaluates the four momentum terms plus a
same-order nonlinear distractor using integration by parts in space and time.

```powershell
$python = 'E:\research\code\miniconda\Asaved\envs\maclearn\python.exe'

# Generate 15 independent 64^3 trajectories (resume-safe).
& $python scripts\generate_data.py

# Discover and independently validate the full momentum equation.
& $python scripts\discover_equation.py
& $python scripts\validate_science.py

# Build the future-energy prediction cache and run a smoke test.
& $python scripts\prepare_cache.py
& $python scripts\diagnose_predictability.py
& $python scripts\run_experiment.py --physics-source discovered --smoke

# Complete 2-region x 3-method x 3-seed corrected experiment.
& $python scripts\run_suite.py

# Reproduce the original manuscript's broader experiment families with v4 data.
& $python scripts\run_baselines.py
& $python scripts\run_relation_ablation.py
& $python scripts\run_noise.py
& $python scripts\discover_noise.py
& $python scripts\validate_extensions.py
```

`run_experiment.py` refuses an artifact whose support, held-out residual,
trajectory bootstrap, support separation or coefficient audit has failed.
Trainable-condition runs are stored under `full_ns_v4_*`.

The expanded learning artifacts are `full_ns_baselines.json`,
`full_ns_relation_ablation.json`, and `full_ns_noise.json` under
`outputs/aggregate/`. The SPIDER-only noise artifact is
`full_ns_spider_noise.json`. They replace the
initial TeX's unsupported baseline, multi-equation, and extreme-noise numbers
with trajectory-disjoint measured results. High-noise SPIDER support is
reported, but artifacts failing coefficient/residual gates are never accepted
for training.

## Repository data policy

The multi-gigabyte HIT arrays, caches, model checkpoints, and per-run histories
are intentionally excluded from Git. The repository retains the compact JSON
files under `outputs/aggregate/` and `outputs/spider/` so that the reported
results and discovered equation remain inspectable. Local files are not deleted
by this policy; `.gitignore` only prevents them from being committed.

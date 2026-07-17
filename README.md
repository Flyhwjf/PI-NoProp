# Physics-Informed NoProp (PI-NoProp)

[GitHub repository](https://github.com/Flyhwjf/PI-NoProp)

This repository contains the HIT-based implementation, SPIDER equation
discovery pipelines, optimized local NoProp training code, experiment artifacts,
and the accompanying paper.  The archived v1 path reproduces the published
single-snapshot spatial relation.  The new v2 path generates trajectory-disjoint
decaying HIT and discovers the complete time-dependent momentum equation before
transferring its measured coefficients into a first-frame physical condition
and strictly local NoProp objectives. The corrected noise-to-label schedule and
equation-conditioned 3-D encoder raise three-seed trajectory-disjoint accuracy
from 20.37% to 81.79% (low enstrophy) and from 19.44% to 73.41% (high
enstrophy).

## Directory guide

| Path | Purpose | Maintenance rule |
| --- | --- | --- |
| `src/` | Models, data loaders, physics losses, and SPIDER implementation | Source code |
| `scripts/` | Current full-NS data, discovery, training, plotting, and validation entry points | Source code |
| `scripts/legacy/` | Preserved v1/single-snapshot experiments and utilities | Historical source; not used by the current paper |
| `tests/` | Optimized-pipeline unit tests | Source code |
| `data/generated/` | Full HIT samples used by discovery | Large required data; do not relocate |
| `data/cache/` | First-frame cache and fixed dataset splits | Regenerable but expensive enough to retain |
| `data/generated_hit_ns_v2/` | Dense, independent decaying-HIT trajectories for full-NS discovery | Regenerable large data; isolated from v1 |
| `data/cache_hit_ns_v2/` | Trajectory-disjoint predictive-learning cache | Regenerable large data |
| `outputs/spider/` | Validated discovered-equation artifacts | Required by SPIDER-informed training |
| `outputs/runs/` | Per-run configurations and metrics | Experiment record |
| `outputs/models/` | Trained model weights | Large experiment artifact |
| `outputs/aggregate/` | Aggregated paper results | Experiment record |
| `paper/` | Current TeX source, figures, and compiled PDF | Current manuscript |
| `初始/` | Original supervisor-provided TeX and reference output | Read-only reference |
| `_archive/` | Superseded project material and unrelated temporary metadata | Not used by current code |

The current code assumes it is launched from the repository root. In particular,
the relative paths `data/cache`, `data/generated`, and `outputs` are part of the
runtime interface and should not be renamed casually.

## Validation

Use the project environment rather than the system Python:

```powershell
$python = 'E:\research\code\miniconda\Asaved\envs\maclearn\python.exe'
& $python scripts\validate_paper.py
& $python scripts\legacy\validate_discovery_pipeline.py
& $python scripts\validate_full_ns_v2.py
& $python -m unittest discover -s tests -v
```

Compile the manuscript from `paper/`:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error "Physics-Informed NoProp.tex"
```

See `OPTIMIZATION.md` for the optimized training design and commands.

## Full Navier--Stokes v2 pipeline

The v2 solver uses correctly normalized 3/2 de-aliasing, RK4 integration and no
unobserved forcing.  Discovery, validation and test sets contain different DNS
trajectories.  The SPIDER library evaluates the four momentum terms plus a
same-order nonlinear distractor using integration by parts in space and time.

```powershell
$python = 'E:\research\code\miniconda\Asaved\envs\maclearn\python.exe'

# Generate 15 independent 64^3 trajectories (resume-safe).
& $python scripts\generate_hit_ns_v2.py

# Discover and independently validate the full momentum equation.
& $python scripts\discover_hit_ns_v2.py
& $python scripts\validate_full_ns_v2.py

# Build the future-energy prediction cache and run a smoke test.
& $python scripts\prepare_hit_ns_v2_cache.py
& $python scripts\diagnose_hit_ns_v2_prediction.py
& $python scripts\run_full_ns_v2.py --physics-source discovered --smoke

# Complete 2-region x 3-method x 3-seed corrected experiment.
& $python scripts\run_full_ns_v2_suite.py

# Reproduce the original manuscript's broader experiment families with v3 data.
& $python scripts\run_full_ns_v3_baselines.py
& $python scripts\run_full_ns_v3_equation_ablation.py
& $python scripts\run_full_ns_v3_noise.py
& $python scripts\discover_hit_ns_v3_noise.py
& $python scripts\validate_full_ns_v3_extensions.py
```

`run_full_ns_v2.py` refuses an artifact whose support, held-out residual,
trajectory bootstrap, support separation or coefficient audit has failed.
Corrected runs are stored under `full_ns_v3_*`; the original v2 run records and
aggregate remain untouched for provenance.

The expanded artifacts are `full_ns_v3_baselines.json`,
`full_ns_v3_equation_ablation.json`, `full_ns_v3_noise.json`, and
`full_ns_v3_spider_noise.json` under `outputs/aggregate/`. They replace the
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

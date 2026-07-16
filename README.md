# Discovery-Informed NoProp

This repository contains the HIT-based implementation, SPIDER spatial-equation
discovery pipeline, optimized local NoProp training code, experiment artifacts,
and the accompanying paper.

## Directory guide

| Path | Purpose | Maintenance rule |
| --- | --- | --- |
| `src/` | Models, data loaders, physics losses, and SPIDER implementation | Source code |
| `scripts/` | Reproducible data, discovery, training, plotting, and validation entry points | Source code |
| `tests/` | Optimized-pipeline unit tests | Source code |
| `data/generated/` | Full HIT samples used by discovery | Large required data; do not relocate |
| `data/cache/` | First-frame cache and fixed dataset splits | Regenerable but expensive enough to retain |
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
& $python scripts\validate_discovery_pipeline.py
& $python -m unittest tests.test_optimized -v
```

Compile the manuscript from `paper/`:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error "Physics-Informed NoProp.tex"
```

See `OPTIMIZATION.md` for the optimized training design and commands.

## Repository data policy

The multi-gigabyte HIT arrays, caches, model checkpoints, and per-run histories
are intentionally excluded from Git. The repository retains the compact JSON
files under `outputs/aggregate/` and `outputs/spider/` so that the reported
results and discovered equation remain inspectable. Local files are not deleted
by this policy; `.gitignore` only prevents them from being committed.

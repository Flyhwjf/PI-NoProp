# Physics-Informed NoProp (PI-NoProp)

[GitHub repository](https://github.com/Flyhwjf/PI-NoProp)

This repository contains the HIT-based implementation, SPIDER equation
discovery pipelines, optimized local NoProp training code, and compact
experiment records. Unpublished manuscripts and editorial materials are
maintained locally. The current path generates trajectory-disjoint
decaying HIT and discovers the complete time-dependent momentum equation before
transferring its measured coefficients into a first-frame physical condition
and strictly local NoProp objectives. The model receives a 32^3 velocity-pressure
context while labels and temporal physics are evaluated on a centred 16^3 cube.
The residual context encoder and cached condition path achieve three-seed
trajectory-disjoint accuracy of 87.35% (low enstrophy) and 81.75% (high
enstrophy) for discovered physics. The final latent is read by cosine
similarity against frozen label embeddings, so the current pipeline has no
separately trained output classifier. The 16^3 v4 artifacts remain available as
the manuscript-compatible historical reference.

## Directory guide

| Path | Purpose | Maintenance rule |
| --- | --- | --- |
| `src/` | Models, data loaders, physics losses, and SPIDER implementation | Source code |
| `scripts/` | Current full-NS data, discovery, training, plotting, and validation entry points | Source code |
| `tests/` | Optimized-pipeline unit tests | Source code |
| `data/generated_hit_ns/` | Dense, independent decaying-HIT trajectories for full-NS discovery | Regenerable large data |
| `data/cache_hit_ns_input32_target16/` | Current 32^3-context/16^3-target trajectory-disjoint learning cache | Regenerable large data |
| `data/cache_hit_ns/` | Legacy 16^3-input predictive-learning cache | Regenerable historical data |
| `outputs/spider/` | Validated discovered-equation artifacts | Required by SPIDER-informed training |
| `outputs/runs/` | Per-run configurations and metrics | Experiment record |
| `outputs/models/` | Trained model weights | Large experiment artifact |
| `outputs/aggregate/` | Aggregated paper results | Experiment record |
| `paper/figure_code/` | Python plotting utilities used by the experiment scripts | Source code |

The current code assumes it is launched from the repository root. In particular,
the relative paths `data/cache_hit_ns_input32_target16`, `data/generated_hit_ns`, and
`outputs` are part of the runtime interface and should not be renamed casually.

## Validation

Use the project environment rather than the system Python:

```powershell
$python = 'E:\research\code\miniconda\Asaved\envs\maclearn\python.exe'
& $python scripts\validate_science.py
& $python -m unittest discover -s tests -v
```

`validate_paper.py` and `validate_extensions.py` additionally check the local
unpublished manuscript and require its TeX source. Manuscript compilation is a
local author workflow; manuscript sources and PDFs are excluded from this
repository.

## Full Navier--Stokes pipeline

The solver in `src/hit_dns.py` uses correctly normalized 3/2 de-aliasing,
RK4 integration and unforced decaying HIT, with automatic divergence, CFL and
energy-decay checks. Discovery, validation and test sets contain different DNS
trajectories (9/3/3), including trajectory-level bootstrap resampling during
equation validation. The SPIDER library in `src/spider_ns.py` evaluates the four
momentum terms plus a same-order nonlinear distractor using integration by parts
in space and time, moving derivatives onto compact test functions.

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

# Complete 2-region x 3-method x 3-seed 32^3-context experiment.
& $python scripts\run_suite.py

# Complete the size-matched 32^3-context/16^3-target extensions.
& $python scripts\run_baselines.py
& $python scripts\run_lambda_ablation.py
& $python scripts\run_relation_ablation.py
& $python scripts\run_decoder_ablation.py
& $python scripts\run_noise.py
& $python scripts\analyze_latents.py

# Discovery noise uses the unchanged physical DNS trajectories.
& $python scripts\discover_noise.py
& $python scripts\plot_results.py --protocol v5
```

`run_experiment.py` refuses an artifact whose support, held-out residual,
trajectory bootstrap, support separation or coefficient audit has failed.
Current prototype-readout runs are stored under
`full_ns_v5_input32_target16_residual_warmstart_*`; the original v4 runs remain
available for historical reproduction. The legacy `--classifier-epochs` option is
retained as an alias for condition-pretraining epochs.

The current learning artifacts under `outputs/aggregate/` share the prefix
`full_ns_v5_input32_target16_residual_warmstart_` and cover main results,
baselines, physics-weight and relation ablations, decoder comparisons,
predictive noise and latent analysis. Their metadata records input geometry,
target geometry, context mode, physics weight and three seeds. The independent
SPIDER DNS-field noise artifact remains `full_ns_spider_noise.json`.
For the local author workflow, `sync_paper_results.py` generates numerical
macros and a source-hashed evidence report from the seven matching learning
aggregates. Generated TeX and figure files remain local.

## Optimized training

The current 32^3-context/16^3-target training design uses the following components:

- The spatial encoder combines a centred 16^3 local backbone with an independent
  32^3 context backbone. A bounded, zero-initialized residual gate controls how
  context modifies the local representation. The context branch is adapted
  after the centre pathway has been pretrained or loaded from matched v4 weights.
- The discovered coefficients combine target-free first-frame convection,
  pressure and viscous energy contributions into a physical condition, which
  is fused with the spatial features.
- `TemporalPhysicsDecoder` reconstructs nine velocity-pressure frames on a
  16^3 grid. `TemporalNSPhysicsLoss` evaluates differentiable weak-form residuals
  using a validated full-momentum artifact. The decoder is frozen during local
  block training, while gradients pass through it to the current block.
- Frozen condition features are cached once. The local stage then loads sample
  indices and labels, avoiding repeated 3-D encoding and 32^3 tensor collation.
- The noise-to-label schedule uses matching noise levels for local training and
  sequential inference.
- The formal suite uses 100 updates per block and three matched seeds, with the
  same architecture, split and budget for no physics, analytic NS and discovered
  NS. Each physical condition has its own frozen shared components.
- The final latent is classified by cosine similarity to frozen label-embedding
  prototypes, preserving the local block objectives without a separate output-head
  training stage.

The current formal results are stored in
`outputs/aggregate/full_ns_v5_input32_target16_residual_warmstart_results.json`.
The earlier 16^3 v4 results remain in `outputs/aggregate/full_ns_results.json`
for historical reproduction. Every formal run writes its configuration,
metrics, history and checkpoint under `outputs/runs/<run_id>`.

The original 16^3 extension artifacts retain their existing filenames and are
kept separately from the current size-specific results. Plotting selects v5
by default; `--protocol old16` explicitly requests historical plots.
Training accepts discovered artifacts that pass the support, coefficient,
residual, separation and bootstrap gates.

## Repository data policy

The repository retains source code, tests, Python plotting utilities, and
compact experimental results. Under `outputs/`, only the direct JSON files in
`aggregate/` and `spider/` are eligible for inclusion so that the measured
results and discovered equation remain inspectable.

Until the manuscript is released, English and Chinese manuscript sources,
final PDFs, bibliography files, rendered figures, reference-verification
records, and revision/comparison notes remain local. In `paper/`, only Python
files directly under `figure_code/` are eligible for inclusion because the
experiment plotting code imports them. `paper_ch/` and `paper-v1/` are excluded
in full; `outputs/literature_revision_sources.json` is also excluded.

The multi-gigabyte HIT arrays, caches, model checkpoints, per-run histories,
revision backups, layout checks, LaTeX intermediates, and figure previews stay
local. Supervisor originals, private copyright-registration documents, the root
project archive and revision PDF, and the standalone HTML drawing also stay
local. `.gitignore` preserves these files on disk and excludes untracked matches
from normal staging. Removing a tracked file from the index excludes it from
subsequent commits while preserving the local copy; earlier commits continue
to contain their original files.

# PI-NoProp optimized training

## Full-NS v2 path

The current research path is the trajectory-disjoint full-Navier--Stokes v2
pipeline. It is isolated from the retained single-snapshot v1 experiment:

- `src/hit_dns_v2.py` implements corrected 3/2 de-aliasing, RK4 and unforced
  decaying HIT with machine-checked divergence, CFL and energy decay.
- `src/spider_ns_v2.py` moves every time and space derivative onto analytic
  compact test functions and selects among four momentum terms plus a
  nonlinear distractor.
- discovery/validation/test use different DNS trajectories (9/3/3), including
  trajectory-level bootstrap resampling.
- `TemporalPhysicsDecoder` reconstructs nine physical frames, and
  `TemporalNSPhysicsLoss` accepts only a validated full-NS artifact.
- the SPIDER coefficients also combine target-free first-frame convection,
  pressure and viscous energy contributions into a predictive condition;
- a derivative-preserving 3-D encoder and a train/inference-consistent
  noise-to-label schedule replace the lossy pooled encoder and reversed chain;
- the formal suite uses 100 updates per block and three matched seeds with the
  same architecture, split and budget for no physics, analytic NS and
  discovered NS. Each physical condition has its own frozen pretraining.

The corrected three-seed results are 81.79%/73.41% for discovered PI-NoProp in
low/high enstrophy, versus 20.37%/19.44% for the no-equation ablation. Compact
results are stored in `outputs/aggregate/full_ns_v3_results.json`; v2 artifacts
are retained as diagnostic provenance.

The restored experiment families add five-method baselines, four clean-trained
noise levels, noisy weak-SPIDER recovery, and three transferred relation sets.
The full NS+pressure-Poisson+energy set reaches 83.02% low-enstrophy accuracy
and lowers the pressure-Poisson/energy residuals to 0.421/0.437. The noise audit
is intentionally fail-closed: only the clean discovered artifact passes every
support, coefficient, residual, separation, and bootstrap gate.

Reproduction commands are listed in `README.md`. The sections below document
the retained v1 spatial-pressure implementation.

The optimized path preserves the original files and uses a separate entry point.

```powershell
conda run -n maclearn python scripts/legacy/check_environment.py
conda run -n maclearn python scripts/legacy/prepare_cache.py
conda run -n maclearn python scripts/legacy/run_fast.py --region centre --seed 42
```

Key differences from the legacy path:

- classification reads a contiguous cache containing frame zero only;
- train/validation/test are fixed at 192/32/32 samples;
- the full 32-frame NPZ files remain available for SPIDER;
- single-snapshot training uses spatial continuity and pressure-Poisson losses;
- each batch updates exactly one independently optimized NoProp block;
- no classifier or physics gradient crosses a block boundary;
- the encoder, label embeddings, and physics decoder are pretrained once and frozen;
- the clean 128 MB regional dataset is cached on the GPU;
- AMP, TF32, and a batch size of 64 are used on the RTX 4060;
- every formal run writes its configuration, metrics, history, and checkpoint under
  `outputs/runs/<run_id>`.

## Validated discovery-to-training path

The current SPIDER path discovers a dimensionally constrained spatial scalar
relation on 60 archived DNS snapshots and validates it on 20 disjoint
snapshots.  It searches the weak-form library
`{pressure_laplacian, convection_divergence, kinetic_energy_laplacian}` and
writes coefficients, residuals, bootstrap stability, data files, grid spacing,
and provenance to `outputs/spider/hit_spatial_equation.json`.

```powershell
conda run -n maclearn python scripts/legacy/discover_hit_spatial.py
conda run -n maclearn python scripts/legacy/run_fast.py --region centre --physics-source discovered
```

`--physics-source discovered` accepts only an artifact whose independent
validation passed.  It never silently falls back to an analytic equation.
The analytic pressure--Poisson prior remains available as a matched ablation
through `--physics-source analytic`; `--physics-source none` is vanilla local
NoProp.  Decoder outputs are converted from cache-standardized values back to
physical units before either residual is evaluated.

Useful commands:

```powershell
# Fast correctness check
conda run -n maclearn python scripts/legacy/run_fast.py --region centre --smoke

# Vanilla local NoProp
conda run -n maclearn python scripts/legacy/run_fast.py --region centre --lambda-phys 0 --no-continuity --no-pressure-poisson

# Pressure-Poisson only
conda run -n maclearn python scripts/legacy/run_fast.py --region centre --no-continuity

# Tests and performance report
conda run -n maclearn python -m unittest tests.test_optimized -v
conda run -n maclearn python scripts/legacy/benchmark_training.py
conda run -n maclearn python scripts/legacy/aggregate_runs.py
```

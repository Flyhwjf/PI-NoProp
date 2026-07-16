# PI-NoProp optimized training

The optimized path preserves the original files and uses a separate entry point.

```powershell
conda run -n maclearn python scripts/check_environment.py
conda run -n maclearn python scripts/prepare_cache.py
conda run -n maclearn python scripts/run_fast.py --region centre --seed 42
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
conda run -n maclearn python scripts/discover_hit_spatial.py
conda run -n maclearn python scripts/run_fast.py --region centre --physics-source discovered
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
conda run -n maclearn python scripts/run_fast.py --region centre --smoke

# Vanilla local NoProp
conda run -n maclearn python scripts/run_fast.py --region centre --lambda-phys 0 --no-continuity --no-pressure-poisson

# Pressure-Poisson only
conda run -n maclearn python scripts/run_fast.py --region centre --no-continuity

# Tests and performance report
conda run -n maclearn python -m unittest tests.test_optimized -v
conda run -n maclearn python scripts/benchmark_training.py
conda run -n maclearn python scripts/aggregate_runs.py
```

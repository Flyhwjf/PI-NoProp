# PI-NoProp optimized training

## Full-NS path

The current research path is the trajectory-disjoint full-Navier--Stokes
pipeline:

- `src/hit_dns.py` implements corrected 3/2 de-aliasing, RK4 and unforced
  decaying HIT with machine-checked divergence, CFL and energy decay.
- `src/spider_ns.py` moves every time and space derivative onto analytic
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

The trainable-condition three-seed results are 86.73%/82.94% for discovered
PI-NoProp in low/high enstrophy, versus 17.59%/17.46% for the no-equation
ablation. Compact results are stored in
`outputs/aggregate/full_ns_results.json`.

The restored experiment families add five-method baselines, four clean-trained
noise levels, noisy weak-SPIDER recovery, and three transferred relation sets.
The full NS+pressure-Poisson+energy set reaches 86.73% low-enstrophy accuracy
and lowers the pressure-Poisson/energy residuals to 0.251/0.351. The noise audit
is intentionally fail-closed: only the clean discovered artifact passes every
support, coefficient, residual, separation, and bootstrap gate.

Reproduction commands are listed in `README.md`. Every formal run writes its
configuration, metrics, history, and checkpoint under `outputs/runs/<run_id>`.

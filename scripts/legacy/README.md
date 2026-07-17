# Legacy single-snapshot pipeline

These scripts preserve the earlier spatial-relation/v1 experiments. They are
kept for provenance and historical reproducibility, but the current manuscript
uses the trajectory-based full-NS scripts in the parent directory.

Run legacy commands from the repository root, for example:

```powershell
python scripts/legacy/run_fast.py --help
python scripts/legacy/validate_discovery_pipeline.py
```

Legacy plotting scripts write to `paper/figures/legacy/` so they cannot
overwrite figures referenced by the current manuscript.

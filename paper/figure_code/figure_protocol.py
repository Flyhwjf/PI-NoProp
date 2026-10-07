"""Data-source selection only; individual figures retain their tuned styles.

The default protocol is 32^3 context / centred 16^3 target, residual warm-start,
lambda=0.1. Legacy 16^3 sources require an explicit ``--protocol old16``.
Missing v5 aggregates never fall back to legacy results. DNS/SPIDER diagnostics
remain tied to the original 64^3 DNS rather than to a learning-cache protocol.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXTENSIONS = (
    "baselines", "noise", "lambda_ablation", "decoder_ablation",
    "relation_ablation", "latent_analysis",
)


@dataclass(frozen=True)
class FigureProtocol:
    name: str
    aggregate_prefix: str
    run_prefix: str
    cache_name: str
    input_size: int
    target_size: int
    lambda_phys: float
    context_mode: str

    def aggregate_path(self, suffix: str) -> Path:
        if suffix not in ("results", *EXTENSIONS):
            raise ValueError(f"Unknown aggregate suffix: {suffix}")
        return ROOT / "outputs/aggregate" / f"{self.aggregate_prefix}_{suffix}.json"

    @property
    def cache_path(self) -> Path:
        return ROOT / "data" / self.cache_name

    def run_path(self, region: str, seed: int, method: str = "discovered") -> Path:
        weight = 0.0 if method == "none" else self.lambda_phys
        token = f"{weight:g}".replace(".", "p")
        return ROOT / "outputs/runs" / (
            f"{self.run_prefix}_{method}_{region}_lambda{token}_seed{seed}"
        )


V5 = FigureProtocol(
    "v5", "full_ns_v5_input32_target16_residual_warmstart",
    "full_ns_v5_input32_target16_residual_warmstart",
    "cache_hit_ns_input32_target16", 32, 16, 0.1, "residual_warmstart",
)
OLD16 = FigureProtocol(
    "old16", "full_ns", "full_ns_v4", "cache_hit_ns", 16, 16, 0.01, "single",
)
PROTOCOLS = {protocol.name: protocol for protocol in (V5, OLD16)}


def parse_cli(default_output: Path, description: str) -> tuple[FigureProtocol, Path]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--protocol", choices=tuple(PROTOCOLS), default="v5",
        help="v5: input32/target16, residual warm-start, lambda=0.1 (default); "
             "old16: explicit legacy input16/target16, lambda=0.01",
    )
    parser.add_argument("--output", type=Path, default=default_output,
                        help="PDF destination; the default preserves the manuscript figure name")
    args = parser.parse_args()
    return PROTOCOLS[args.protocol], args.output


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing selected-protocol source: {path}\n"
            "No legacy fallback is used; supply this source or explicitly select --protocol old16."
        )
    print(f"Source: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_metadata(protocol: FigureProtocol, metadata: dict, source: Path,
                      *, check_lambda: bool = True) -> None:
    """Reject contradictory provenance without imposing a new extension schema."""
    expected = {
        "input_spatial_size": protocol.input_size,
        "target_spatial_size": protocol.target_size,
        "spatial_context_mode": protocol.context_mode,
    }
    for key, value in expected.items():
        if key in metadata and metadata[key] != value:
            raise ValueError(f"{source}: {key}={metadata[key]!r}, expected {value!r}")
    if check_lambda:
        for key in ("lambda_phys", "lambda_weight"):
            if key in metadata and not math.isclose(
                float(metadata[key]), protocol.lambda_phys, rel_tol=1e-9, abs_tol=1e-12
            ):
                raise ValueError(f"{source}: {key} must be {protocol.lambda_phys:g}")
    revision = metadata.get("model_revision", "")
    if protocol.name == "v5" and revision == (
        "trainable-physics-condition-fusion-prototype-readout"
    ):
        raise ValueError(f"{source}: legacy v4 model_revision in a v5 source")


def load_aggregate(protocol: FigureProtocol, suffix: str) -> dict:
    path = protocol.aggregate_path(suffix)
    artifact = read_json(path)
    validate_metadata(protocol, artifact.get("protocol", {}), path,
                      check_lambda=suffix != "lambda_ablation")
    return artifact


def main() -> None:
    """Read-only inventory of the selected protocol's expected aggregates."""
    protocol, _ = parse_cli(ROOT / "paper/figures", __doc__)
    for suffix in ("results", *EXTENSIONS):
        path = protocol.aggregate_path(suffix)
        print(f"{'available' if path.is_file() else 'MISSING'}: {path}")


if __name__ == "__main__":
    main()

"""Generate local-loss curves and final prototype-readout diagnostics.

Left: actual per-block losses from the selected runs' history.npz files,
normalised by each block's first-ten-update median, then median and IQR across
blocks/runs. Right: final train/validation prototype accuracy, mean +/- sample
SD across three seeds per enstrophy group, from the same runs' metrics.json.
There is no final classifier training stage and no invented adaptation curve.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure_protocol import (
    FigureProtocol, V5, load_aggregate, parse_cli, read_json, validate_metadata,
)


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "paper/figures"
OUTPUT = FIGURES / "fig_training_convergence.pdf"
COLORS = {
    "discovered": "#2b7bbb",
    "green": "#8196a8",
    "navy": "#274c77",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10.5,
            "axes.labelsize": 11.5,
            "axes.titlesize": 12.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "figure.dpi": 150,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def box_axes(axis: plt.Axes) -> None:
    """Use a visible, consistent four-sided frame for each panel."""
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#334155")
        spine.set_linewidth(0.85)


def place_panel_label_left_of_title(
    fig: plt.Figure, axis: plt.Axes, label: str
) -> None:
    """Place a panel marker immediately to the left of the centered title."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title_bbox = axis.title.get_window_extent(renderer=renderer)
    marker = axis.text(
        0,
        0,
        label,
        transform=axis.transAxes,
        ha="right",
        va="center",
        fontsize=12,
        fontweight="bold",
        clip_on=False,
    )
    marker.set_in_layout(False)
    fig.canvas.draw()
    marker_bbox = marker.get_window_extent(renderer=renderer)
    x_px = title_bbox.x0 - marker_bbox.width - 5
    y_px = title_bbox.y0 + 0.5 * title_bbox.height
    x_axes, y_axes = axis.transAxes.inverted().transform((x_px, y_px))
    marker.set_position((x_axes, y_axes))


def load_diagnostics(protocol: FigureProtocol = V5) -> tuple[dict, dict, int]:
    """Validate all selected histories and metrics before any figure is written."""
    aggregate = load_aggregate(protocol, "results")
    regions = ("low_enstrophy", "high_enstrophy")
    seeds = aggregate["protocol"]["seeds"]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("The readout diagnostic requires three distinct seeds per group")
    steps = int(aggregate["protocol"]["steps_per_block"])
    if steps <= 0:
        raise ValueError("steps_per_block must be positive")
    local_curves = {key: [] for key in ("loss", "diff", "phys")}
    readout = {region: {"train": [], "val": []} for region in regions}

    for region in regions:
        for seed in seeds:
            run_dir = protocol.run_path(region, seed)
            # v5 stores real directory names. The legacy v4 summary stores
            # method:seed display tags instead, so old16 uses validated config
            # and metrics identities below rather than treating tags as paths.
            if protocol.name == "v5" and run_dir.name not in (
                aggregate["results"][region]["discovered"]["run_ids"]
            ):
                raise ValueError(f"Run not listed in the selected main aggregate: {run_dir}")
            path = run_dir / "history.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Missing selected-protocol history (no fallback): {path}")
            metrics_path = run_dir / "metrics.json"
            metrics = read_json(metrics_path)
            validate_metadata(protocol, metrics, metrics_path)
            config_path = run_dir / "config.json"
            config = read_json(config_path)
            validate_metadata(protocol, {
                "input_spatial_size": config["data"]["subdomain_size"],
                "target_spatial_size": config["data"].get(
                    "target_subdomain_size") or config["data"]["subdomain_size"],
                "spatial_context_mode": config["noprop"].get("spatial_context_mode", "single"),
                "lambda_weight": config["physics"]["lambda_weight"],
            }, config_path)
            blocks = int(config["diffusion"]["T"])
            if config["training"]["local_steps_per_block"] != steps:
                raise ValueError(f"Local-step count differs from main aggregate: {config_path}")
            if (metrics.get("region"), metrics.get("seed"), metrics.get("physics_source")) != (
                region, seed, "discovered"
            ):
                raise ValueError(f"Mismatched run identity: {metrics_path}")
            with np.load(path, allow_pickle=True) as history:
                local = history["local"].tolist()
                if "classifier" in history and len(history["classifier"]):
                    raise ValueError(f"Unexpected trained-classifier stage: {path}")
            print(f"Source: {path} (local losses only)")
            if len(local) != blocks * steps or metrics.get("readout") != (
                "cosine similarity to frozen label embeddings"
            ):
                raise ValueError(f"Incomplete or non-prototype history: {path}")
            if metrics["block_updates"] != [steps] * blocks:
                raise ValueError(f"Incomplete block updates: {metrics_path}")

            for block in range(blocks):
                sequence = [record for record in local if record["block"] == block]
                if len(sequence) != steps:
                    raise ValueError(f"Block {block} is incomplete in {path}")
                for key in local_curves:
                    values = np.asarray(
                        [record[key] for record in sequence], dtype=float
                    )
                    if not np.all(np.isfinite(values)) or np.any(values < 0):
                        raise ValueError(f"Invalid {key} values in block {block}: {path}")
                    reference = np.median(values[:10])
                    local_curves[key].append(values / max(reference, 1e-12))

            for split, key in (("train", "train_readout"), ("val", "validation_readout")):
                accuracy = float(metrics[key]["accuracy"])
                if not np.isfinite(accuracy) or not 0 <= accuracy <= 100:
                    raise ValueError(f"Invalid {key} accuracy: {metrics_path}")
                readout[region][split].append(accuracy)

    return local_curves, readout, steps


def plot(protocol: FigureProtocol = V5, output: Path = OUTPUT) -> None:
    configure_style()
    local_curves, readout, steps = load_diagnostics(protocol)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.9, 3.2),
        gridspec_kw={"width_ratios": (1, 1)},
        constrained_layout=True,
    )
    for axis in axes:
        box_axes(axis)

    ax = axes[0]
    updates = np.arange(1, steps + 1)
    curve_styles = (
        ("loss", COLORS["navy"], "Total"),
        ("diff", COLORS["discovered"], "Denoising"),
        ("phys", COLORS["green"], "Physics"),
    )
    for key, colour, label in curve_styles:
        values = np.asarray(local_curves[key])
        median = np.median(values, axis=0)
        lower, upper = np.quantile(values, [0.25, 0.75], axis=0)
        ax.fill_between(updates, lower, upper, color=colour, alpha=0.14, linewidth=0)
        ax.plot(updates, median, color=colour, linewidth=1.5, label=label,
                linestyle={"loss": "-", "diff": "--", "phys": "-."}[key])
    ax.set_yscale("log")
    ax.set(
        xlabel="Local updates per block",
        ylabel="Normalised loss",
        title="Local block optimisation",
    )
    ax.grid(axis="y", which="both", alpha=0.20)
    ax.grid(axis="x", alpha=0.12)
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#94a3b8",
        framealpha=0.92,
        fancybox=False,
        fontsize=10.0,
        loc="lower left",
    )
    ax = axes[1]
    positions = {"low_enstrophy": 0, "high_enstrophy": 1}
    for region, colour, label in (
        ("low_enstrophy", "#6baed6", "Low enstrophy"),
        ("high_enstrophy", "#2171b5", "High enstrophy"),
    ):
        train = np.asarray(readout[region]["train"], dtype=float)
        validation = np.asarray(readout[region]["val"], dtype=float)
        x = positions[region]
        offset = 0.15
        width = 0.26
        ax.bar(
            x - offset,
            train.mean(),
            width,
            yerr=train.std(ddof=1),
            color=colour,
            alpha=0.55,
            edgecolor="#475569",
            linewidth=0.45,
            error_kw={"elinewidth": 1.25},
            capsize=3,
            label="Train" if region == "low_enstrophy" else "_nolegend_",
            zorder=3,
        )
        ax.bar(
            x + offset,
            validation.mean(),
            width,
            yerr=validation.std(ddof=1),
            color=colour,
            edgecolor="#475569",
            linewidth=0.45,
            error_kw={"elinewidth": 1.8},
            capsize=3,
            hatch="//",
            label="Validation" if region == "low_enstrophy" else "_nolegend_",
            zorder=3,
        )
    ax.axhline(
        20,
        color="#64748b",
        linestyle=":",
        linewidth=1.2,
    )
    ax.set(
        xlabel="Enstrophy group",
        ylabel="Accuracy (%)",
        xticks=[0, 1],
        xticklabels=["Low", "High"],
        ylim=(0, 101),
        xlim=(-0.55, 1.55),
        title="Frozen prototype readout",
    )
    ax.grid(axis="y", alpha=0.20)
    ax.grid(axis="x", alpha=0.12)
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#94a3b8",
        framealpha=0.92,
        fancybox=False,
        fontsize=10.0,
        ncol=2,
        loc="lower right",
    )

    place_panel_label_left_of_title(fig, axes[0], "(a)")
    place_panel_label_left_of_title(fig, axes[1], "(b)")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Right panel: final train/validation prototype readout, mean +/- sample SD; "
          "three seeds per group, not a classifier/adaptation convergence curve.")
    print(f"Wrote {output}")


if __name__ == "__main__":
    plot(*parse_cli(OUTPUT, __doc__))

"""Prediction noise sweep: accuracy and decoded momentum residual.

Source: the explicitly selected figure protocol's noise aggregate.
Error bars are complete sample SDs over model seeds, after averaging the
observation-noise repetitions within each seed. No training is performed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from figure_protocol import FigureProtocol, V5, load_aggregate, parse_cli


ROOT = Path(__file__).resolve().parents[2]
AGGREGATE = V5.aggregate_path("noise")
OUTPUT = ROOT / "paper/figures/fig_noise.pdf"
REGIONS = ("low_enstrophy", "high_enstrophy")
METHODS = ("none", "discovered")
REGION_STYLES = (
    ("low_enstrophy", "#4D87AD", "o", "low"),
    ("high_enstrophy", "#244D70", "s", "high"),
)


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10.2,
        "axes.labelsize": 10.5,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 10.0,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.grid": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": None,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def extract_noise_data(artifact: dict) -> tuple[np.ndarray, dict]:
    """Keep the source's numeric levels, means and full SDs, in fixed order."""
    levels = np.asarray(
        artifact["protocol"]["levels_in_channel_standard_deviations"], dtype=float
    )
    if (levels.ndim != 1 or len(levels) < 3 or not np.isfinite(levels).all()
            or np.any(levels < 0) or np.any(np.diff(levels) <= 0)):
        raise ValueError("Noise levels must be finite, nonnegative and increasing")
    series = {}
    for region in REGIONS:
        for method in METHODS:
            records = artifact["results"][region][method]
            values = {}
            for metric in ("accuracy", "eta_ns"):
                for statistic in ("mean", "std"):
                    array = np.asarray([
                        records[str(float(level))][metric][statistic]
                        for level in levels
                    ], dtype=float)
                    if not np.isfinite(array).all():
                        raise ValueError(f"Nonfinite {region}/{method}/{metric}/{statistic}")
                    if np.any(array < 0):
                        raise ValueError(f"Negative {region}/{method}/{metric}/{statistic}")
                    if metric == "accuracy" and statistic == "mean" and np.any(array > 100):
                        raise ValueError("Accuracy means must be percentages in [0, 100]")
                    values[f"{metric}_{statistic}"] = array
            series[region, method] = values
    return levels, series


def figure_caption() -> str:
    return (
        "Clean-trained prediction under additive observation noise. "
        "(a) Five-class test accuracy; the dotted reference is 20% chance accuracy. "
        "(b) Contribution-normalised momentum residual of decoded fields. "
        "Blue shades and circle/square markers distinguish low/high enstrophy; "
        "solid filled markers denote PI-NoProp and dashed open markers the "
        "no-equation control. Points are three-seed means and error bars show "
        "one complete sample standard deviation, after averaging five "
        "observation-noise realisations within each seed. The horizontal axis "
        "uses numerical noise levels on a symmetric-log scale, linear from "
        "zero to 1% of channel standard deviation."
    )


def build_figure(artifact: dict, protocol: FigureProtocol = V5) -> plt.Figure:
    """Construct the two panels without saving or running an experiment."""
    levels, series = extract_noise_data(artifact)
    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.45))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.215, top=0.875, wspace=0.30)
    handles = []
    for region, colour, marker, region_label in REGION_STYLES:
        for method in METHODS:
            filled = method == "discovered"
            linestyle = "-" if filled else (0, (4, 2))
            name = "PI-NoProp" if filled else "No eq."
            record = series[region, method]
            for axis, metric in zip(axes, ("accuracy", "eta_ns")):
                axis.errorbar(
                    levels, record[f"{metric}_mean"],
                    yerr=record[f"{metric}_std"],
                    color=colour, linestyle=linestyle,
                    linewidth=1.6 if filled else 1.25,
                    marker=marker, markersize=4.5,
                    markerfacecolor=colour if filled else "white",
                    markeredgecolor=colour, markeredgewidth=0.9,
                    elinewidth=0.65, capsize=1.8, capthick=0.65, zorder=3,
                )
            handles.append(Line2D(
                [], [], color=colour, linestyle=linestyle, linewidth=1.4,
                marker=marker, markersize=4.5,
                markerfacecolor=colour if filled else "white",
                markeredgecolor=colour, label=f"{name}, {region_label}",
            ))

    axes[0].axhline(20, color="#77838E", linestyle=":", linewidth=0.9, zorder=1)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[1].set_ylabel(r"Momentum residual $\eta_{\mathrm{NS}}$")
    for axis, metric in zip(axes, ("accuracy", "eta_ns")):
        # Include all mean +/- SD endpoints, including negative lower bounds.
        lower = min(float(np.min(row[f"{metric}_mean"] - row[f"{metric}_std"]))
                    for row in series.values())
        upper = max(float(np.max(row[f"{metric}_mean"] + row[f"{metric}_std"]))
                    for row in series.values())
        floor = min(0.0, lower)
        span = max(upper - floor, 1e-6)
        # Reserve a data-free upper area for the inside, two-row legend.
        axis.set_ylim(floor - 0.08 * span if floor < 0 else 0, upper + 0.30 * span)
        axis.set_xscale("symlog", linthresh=0.01, linscale=0.8, base=10)
        axis.set_xlim(-0.0015, float(levels[-1]) * 1.14)
        axis.set_xticks(levels, [f"{100 * level:g}" for level in levels])
        axis.set_xlabel("Observation noise / channel SD (%)", labelpad=5)
        axis.minorticks_off()
        axis.tick_params(length=3, width=0.7)
        axis.grid(axis="y", color="#E1E6EA", linewidth=0.55, zorder=0)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("#566575")
            spine.set_linewidth(0.75)

    axes[0].set_title("(a)  Predictive robustness", loc="left", pad=9)
    axes[1].set_title("(b)  Decoded momentum consistency", loc="left", pad=9)
    legend_kwargs = dict(
        handles=handles, ncol=2, frameon=True, fancybox=False, framealpha=0.85,
        facecolor="white", edgecolor="#93A1AE", fontsize=10.0,
        handlelength=1.3, handletextpad=0.35, columnspacing=0.6,
        borderpad=0.35, labelspacing=0.35,
    )
    for axis in axes:
        axis.legend(loc="upper center", bbox_to_anchor=(0.50, 0.985), **legend_kwargs)
    return fig


def plot(protocol: FigureProtocol = V5, output: Path = OUTPUT) -> None:
    artifact = load_aggregate(protocol, "noise")
    fig = build_figure(artifact, protocol)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    preview = output.with_name(f"{output.stem}_preview.png")
    try:
        fig.savefig(output, format="pdf", dpi=300, facecolor="white",
                    metadata={"Title": "Predictive robustness and physical consistency",
                              "Subject": figure_caption()})
        fig.savefig(preview, format="png", dpi=300, facecolor="white")
    finally:
        plt.close(fig)
    print(f"Wrote {output}\nPreview: {preview}")


if __name__ == "__main__":
    plot(*parse_cli(OUTPUT, __doc__))

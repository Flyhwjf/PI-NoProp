"""Quantitative held-out latent statistics in the original 128-D space.

Metrics are pooled over the three model seeds, not seed-wise estimates.
No PCA/t-SNE coordinates, confidence intervals or error bars are inferred.
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
ARTIFACT = V5.aggregate_path("latent_analysis")
OUTPUT = ROOT / "paper/figures/fig_latent_metrics.pdf"
REGIONS = ("low_enstrophy", "high_enstrophy")
METHODS = ("none", "analytic", "discovered")
METHOD_LABELS = ("No equation", "Analytic NS", "PI-NoProp")
METRICS = (
    "between_class_centroid_distance", "silhouette_128d", "within_class_trace",
)


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10.2,
        "axes.labelsize": 10.2,
        "axes.titlesize": 10.2,
        "xtick.labelsize": 9.8,
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


def extract_metric_matrix(artifact: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (metric x region x method values, region x method counts).

    Explicit key order prevents JSON dictionary order from changing labels.
    Only original-space statistics are accessed; projection points are unused.
    """
    values = np.empty((len(METRICS), len(REGIONS), len(METHODS)), dtype=float)
    counts = np.empty((len(REGIONS), len(METHODS)), dtype=int)
    for region_index, region in enumerate(REGIONS):
        for method_index, method in enumerate(METHODS):
            record = artifact["results"][region][method]
            count = record["n_samples"]
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError("Latent sample counts must be positive integers")
            counts[region_index, method_index] = count
            for metric_index, metric in enumerate(METRICS):
                values[metric_index, region_index, method_index] = float(record[metric])
    if not np.isfinite(values).all():
        raise ValueError("Latent metrics must be finite")
    if np.any(values[0] < 0) or np.any(values[2] < 0):
        raise ValueError("Centroid distances and covariance traces must be nonnegative")
    if np.any(np.abs(values[1]) > 1):
        raise ValueError("Silhouette scores must lie in [-1, 1]")
    if any(len(set(row)) != 1 for row in counts):
        raise ValueError("Compared methods must use the same held-out sample count")
    return values, counts


def figure_caption(artifact: dict) -> str:
    _, counts = extract_metric_matrix(artifact)
    return (
        "Held-out latent statistics in the original 128-dimensional space. "
        "(a) Mean pairwise distance between class centres. "
        "(b) Silhouette score. (c) Mean within-class covariance trace. "
        "Each panel compares the no-equation control, Analytic NS and PI-NoProp "
        "for low- and high-enstrophy groups. Statistics pool latent vectors "
        "from three model seeds, with "
        f"{counts[0, 0]} low-enstrophy and {counts[1, 0]} high-enstrophy "
        "vectors per method. Points are pooled statistics without seed-wise "
        "error bars; no projected coordinates enter these measurements."
    )


def build_figure(artifact: dict) -> plt.Figure:
    values, _ = extract_metric_matrix(artifact)
    configure_style()
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 3.25), sharey=True)
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.24, top=0.87, wspace=0.20)
    y = np.asarray([2.0, 1.0, 0.0])
    styles = (
        ("#4D87AD", "o", 0.115, "Low enstrophy"),
        ("#244D70", "s", -0.115, "High enstrophy"),
    )
    titles = ("(a)  Class-centre distance", "(b)  Silhouette score",
              "(c)  Within-class spread")
    xlabels = ("Mean distance\n(latent units)", "Silhouette\n(dimensionless)",
               "Covariance trace\n(latent units squared)")
    handles = [
        Line2D([], [], marker=marker, color=colour, linestyle="none",
               markersize=5.0, label=label)
        for colour, marker, _, label in styles
    ]
    for index, axis in enumerate(axes):
        for region_index, (colour, marker, offset, _) in enumerate(styles):
            axis.scatter(values[index, region_index], y + offset,
                         marker=marker, s=29, color=colour,
                         edgecolors="white", linewidths=0.35, zorder=3, clip_on=False)
        axis.set_yticks(y)
        axis.set_ylim(-0.45, 2.9)
        axis.set_title(titles[index], loc="left", pad=9)
        axis.set_xlabel(xlabels[index], labelpad=6)
        axis.tick_params(length=3, width=0.7)
        axis.grid(axis="x", color="#E1E6EA", linewidth=0.55, zorder=0)
        axis.grid(axis="y", visible=False)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("#566575")
            spine.set_linewidth(0.75)
        axis.legend(handles=handles, loc="upper right",
                    frameon=True, fancybox=False, framealpha=0.85,
                    facecolor="white", edgecolor="#93A1AE", borderpad=0.3,
                    handlelength=0.9, handletextpad=0.4, labelspacing=0.25)

    axes[0].set_yticklabels(METHOD_LABELS)
    axes[0].set_ylabel("Model variant", labelpad=6)
    axes[0].set_xlim(0, max(10.0, float(values[0].max()) * 1.15))
    axes[0].set_xticks([0, 2, 4, 6, 8, 10])
    low, high = float(values[1].min()), float(values[1].max())
    axes[1].set_xlim(min(-0.06, low - 0.012), max(0.12, high + 0.02))
    axes[1].set_xticks([-0.05, 0, 0.05, 0.10], ["-0.05", "0", "0.05", "0.10"])
    axes[1].axvline(0, color="#7B8793", linestyle=":", linewidth=0.9, zorder=1)
    # A zero origin makes modest covariance differences visually proportional.
    axes[2].set_xlim(0, max(105.0, float(values[2].max()) * 1.12))
    axes[2].set_xticks([0, 25, 50, 75, 100])
    return fig


def plot(protocol: FigureProtocol = V5, output: Path = OUTPUT) -> None:
    artifact = load_aggregate(protocol, "latent_analysis")
    fig = build_figure(artifact)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    preview = output.with_name(f"{output.stem}_preview.png")
    try:
        fig.savefig(output, format="pdf", dpi=300, facecolor="white",
                    metadata={"Title": "Original-space latent metrics",
                              "Subject": figure_caption(artifact)})
        fig.savefig(preview, format="png", dpi=300, facecolor="white")
    finally:
        plt.close(fig)
    print(f"Wrote {output}\nPreview: {preview}")


if __name__ == "__main__":
    plot(*parse_cli(OUTPUT, __doc__))

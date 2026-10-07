"""Generate the held-out latent-representation analysis figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from figure_protocol import FigureProtocol, V5, load_aggregate, parse_cli


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = V5.aggregate_path("latent_analysis")
OUTPUT = ROOT / "paper/figures/fig_latent_analysis.pdf"


METHODS = (
    ("none", "No equation", "#94A3B8"),
    ("analytic", "Analytic", "#5B9BD5"),
    ("discovered", "PI-NoProp", "#1F4E79"),
)
MARKERS = ("o", "s", "^", "D", "P")


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10.0,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.0,
            "xtick.labelsize": 9.4,
            "ytick.labelsize": 9.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "figure.dpi": 160,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def box_axes(axis: plt.Axes) -> None:
    """Use a visible four-sided frame for every manuscript panel."""
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#334155")
        spine.set_linewidth(0.8)


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
        fontsize=11.0,
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


def projection_limits(artifact: dict, projection: str) -> tuple[float, float, float, float]:
    """Return common limits for the two enstrophy panels in one projection row."""
    values = []
    for region in ("low_enstrophy", "high_enstrophy"):
        for method, _, _ in METHODS:
            values.append(np.asarray(artifact["points"][region][method][projection]))
    points = np.concatenate(values, axis=0)
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    span = np.maximum(upper - lower, 1e-8)
    padding = 0.055 * span
    return (
        float(lower[0] - padding[0]),
        float(upper[0] + padding[0]),
        float(lower[1] - padding[1]),
        float(upper[1] + padding[1]),
    )


def plot(protocol: FigureProtocol = V5, output: Path = OUTPUT) -> None:
    configure_style()
    artifact = load_aggregate(protocol, "latent_analysis")

    fig, axes = plt.subplots(2, 2, figsize=(6.9, 6.0))
    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.085,
        top=0.90,
        wspace=0.32,
        hspace=0.38,
    )

    row_specs = (
        ("pca", "PCA", ("PC 1", "PC 2")),
        ("tsne", "t-SNE", ("t-SNE 1", "t-SNE 2")),
    )
    regions = (("low_enstrophy", "Low enstrophy"), ("high_enstrophy", "High enstrophy"))
    limits = {projection: projection_limits(artifact, projection) for projection, _, _ in row_specs}

    for row, (projection, projection_title, axis_labels) in enumerate(row_specs):
        xmin, xmax, ymin, ymax = limits[projection]
        for col, (region, region_title) in enumerate(regions):
            axis = axes[row, col]
            for method, _, colour in METHODS:
                points = artifact["points"][region][method]
                xy = np.asarray(points[projection], dtype=float)
                labels = np.asarray(points["labels"], dtype=int)
                for class_id, marker in enumerate(MARKERS):
                    mask = labels == class_id
                    if not np.any(mask):
                        continue
                    axis.scatter(
                        xy[mask, 0],
                        xy[mask, 1],
                        s=9.0,
                        alpha=0.72,
                        color=colour,
                        marker=marker,
                        edgecolors="white",
                        linewidths=0.25,
                        rasterized=True,
                        zorder=3,
                    )

            axis.set_xlim(xmin, xmax)
            axis.set_ylim(ymin, ymax)
            axis.set_xlabel(axis_labels[0])
            axis.set_ylabel(axis_labels[1])
            axis.set_title(f"{projection_title}: {region_title.split()[0]}", pad=9)
            axis.grid(alpha=0.18)
            axis.tick_params(axis="both", labelsize=9.4)
            box_axes(axis)

    method_handles = [
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="o",
            markersize=6.2,
            markerfacecolor=colour,
            markeredgecolor=colour,
            label=label,
        )
        for _, label, colour in METHODS
    ]
    class_handles = [
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker=marker,
            markersize=6.0,
            markerfacecolor="#475569",
            markeredgecolor="#475569",
            label=f"Class {class_id + 1}",
        )
        for class_id, marker in enumerate(MARKERS)
    ]

    legend_kwargs = {
        "frameon": True,
        "fancybox": False,
        "framealpha": 0.96,
        "edgecolor": "#9CA3AF",
        "facecolor": "white",
        "fontsize": 9.0,
        "handlelength": 1.0,
        "borderpad": 0.45,
        "columnspacing": 1.0,
    }
    class_labels = [
        f"Class {class_id + 1}" for class_id in range(len(MARKERS))
    ]
    combined_handles = method_handles + class_handles
    combined_labels = [label for _, label, _ in METHODS] + class_labels
    compact_legend_kwargs = {
        **legend_kwargs,
        "fontsize": 8.6,
        "framealpha": 0.90,
        "borderpad": 0.30,
        "handlelength": 0.9,
        "handletextpad": 0.35,
        "columnspacing": 0.60,
        "labelspacing": 0.25,
    }
    for axis in axes.flat:
        axis.legend(
            handles=combined_handles,
            labels=combined_labels,
            loc="lower right",
            bbox_to_anchor=(0.985, 0.015),
            ncol=2,
            **compact_legend_kwargs,
        )

    for index, axis in enumerate(axes.flat):
        place_panel_label_left_of_title(fig, axis, f"({chr(97 + index)})")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    plot(*parse_cli(OUTPUT, __doc__))

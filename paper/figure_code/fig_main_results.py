"""Generate the architecture-matched main-results figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure_protocol import FigureProtocol, V5, load_aggregate, parse_cli


ROOT = Path(__file__).resolve().parents[2]
AGGREGATE = V5.aggregate_path("results")
OUTPUT = ROOT / "paper/figures/fig_main_results.pdf"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10.0,
            "axes.labelsize": 12.0,
            "axes.titlesize": 11.8,
            "xtick.labelsize": 10.2,
            "ytick.labelsize": 10.2,
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
    """Use a visible, consistent four-sided frame for each panel."""
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
        fontsize=11.5,
        fontweight="bold",
        clip_on=False,
    )
    marker.set_in_layout(False)
    fig.canvas.draw()
    marker_bbox = marker.get_window_extent(renderer=renderer)
    x_px = title_bbox.x0 - marker_bbox.width - 4
    y_px = title_bbox.y0 + 0.5 * title_bbox.height
    x_axes, y_axes = axis.transAxes.inverted().transform((x_px, y_px))
    marker.set_position((x_axes, y_axes))


def plot(protocol: FigureProtocol = V5, output: Path = OUTPUT) -> None:
    configure_style()
    aggregate = load_aggregate(protocol, "results")
    regions = ("low_enstrophy", "high_enstrophy")
    methods = ("none", "analytic", "discovered")
    labels = ("No physics", "Analytic NS", "Discovered NS")
    region_styles = (
        ("low_enstrophy", "#6baed6", "o", "Low enstrophy"),
        ("high_enstrophy", "#2171b5", "s", "High enstrophy"),
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11.2, 3.65),
        constrained_layout=True,
    )
    for axis in axes:
        box_axes(axis)

    x = np.arange(3)
    for axis, metric, title in zip(
        axes,
        ("accuracy", "eta_ns", "eta_div"),
        ("Future-decay classification",
         rf"Full-NS consistency ($\lambda={protocol.lambda_phys:g}$)",
         "Continuity consistency"),
    ):
        for region, colour, marker, label in region_styles:
            means = np.asarray(
                [
                    aggregate["results"][region][method][metric]["mean"]
                    for method in methods
                ]
            )
            stds = np.asarray(
                [
                    aggregate["results"][region][method][metric]["std"]
                    for method in methods
                ]
            )
            axis.errorbar(
                x,
                means,
                yerr=stds,
                color=colour,
                marker=marker,
                markersize=6.5,
                linewidth=1.8,
                capsize=4,
                label=label,
                zorder=3,
            )
        axis.set_xticks(x, labels, rotation=15)
        axis.set_xlim(-0.35, 2.35)
        axis.set_title(title, pad=11)
        axis.grid(axis="y", alpha=0.20)
        axis.grid(axis="x", visible=False)

    axes[0].axhline(
        20,
        color="#64748b",
        linestyle=":",
        linewidth=1.2,
        label="Five-class chance",
    )
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_ylim(8, 94)
    axes[0].legend(
        frameon=True,
        facecolor="white",
        edgecolor="#94a3b8",
        framealpha=0.92,
        fancybox=False,
        fontsize=8.4,
        loc="lower right",
    )

    axes[1].set_ylabel(r"$\eta_{\mathrm{NS}}$ (lower is better)")
    axes[1].set_ylim(0, 1.08)
    axes[1].legend(
        frameon=True,
        facecolor="white",
        edgecolor="#94a3b8",
        framealpha=0.92,
        fancybox=False,
        fontsize=8.4,
        loc="lower right",
    )
    axes[2].set_ylabel(r"$\eta_{\mathrm{div}}$ (lower is better)")
    axes[2].set_ylim(0, 0.63)
    axes[2].legend(
        frameon=True,
        facecolor="white",
        edgecolor="#94a3b8",
        framealpha=0.92,
        fancybox=False,
        fontsize=8.4,
        loc="lower right",
    )

    # Keep the tuned scale unless new error bars would be clipped by it.
    div_upper = max(
        aggregate["results"][region][method]["eta_div"]["mean"]
        + aggregate["results"][region][method]["eta_div"]["std"]
        for region in regions for method in methods
    )
    axes[2].set_ylim(0, max(0.63, 1.05 * div_upper))

    for index, axis in enumerate(axes):
        place_panel_label_left_of_title(fig, axis, f"({chr(97 + index)})")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    plot(*parse_cli(OUTPUT, __doc__))

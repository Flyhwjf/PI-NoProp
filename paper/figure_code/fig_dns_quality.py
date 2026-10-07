"""Generate the DNS quality-audit figure used by the manuscript.

The figure combines time-window dynamics (left) with trajectory-level
reference-scaled quality diagnostics (right) in a one-row, two-column layout.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/generated_hit_ns/manifest.json"
OUTPUT = ROOT / "paper/figures/fig_dns_quality.pdf"

COLORS = {
    "energy": "#245a8d",
    "dissipation": "#657b91",
    "reynolds": "#9bb5ca",
    "cfl": "#3b82b6",
    "divergence": "#8196a8",
    "limit": "#475569",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Times"],
            "mathtext.fontset": "stix",
            "font.size": 11.0,
            "axes.labelsize": 11.5,
            "axes.titlesize": 12.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "grid.linewidth": 0.7,
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
        spine.set_linewidth(0.85)


def place_panel_label_left_of_title(fig, axis: plt.Axes, label: str) -> None:
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
    x_px = title_bbox.x0 - marker_bbox.width - 5
    y_px = title_bbox.y0 + 0.5 * title_bbox.height
    x_axes, y_axes = axis.transAxes.inverted().transform((x_px, y_px))
    marker.set_position((x_axes, y_axes))


def normalised_series(diagnostics: list[list[dict]], name: str) -> np.ndarray:
    values = np.asarray(
        [[item[name] for item in trajectory] for trajectory in diagnostics],
        dtype=float,
    )
    return values / values[:, :1]


def add_split_boundaries(ax: plt.Axes) -> None:
    for boundary in (8.5, 11.5):
        ax.axvline(
            boundary,
            color="#94a3b8",
            linewidth=0.75,
            linestyle=(0, (2, 2)),
            alpha=0.65,
            zorder=1,
        )


def plot() -> None:
    configure_style()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = sorted(manifest["trajectories"], key=lambda item: item["trajectory_id"])
    diagnostics = [record["diagnostics"] for record in records]

    time = np.asarray([item["time"] for item in diagnostics[0]], dtype=float)
    time -= time[0]
    energy = normalised_series(diagnostics, "kinetic_energy")
    dissipation = normalised_series(diagnostics, "dissipation")
    reynolds = normalised_series(diagnostics, "re_lambda")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.9, 3.3),
        gridspec_kw={"width_ratios": (1.0, 1.0)},
        constrained_layout=True,
    )
    for axis in axes:
        box_axes(axis)

    # (a) Time-window dynamics.
    ax = axes[0]
    series = {
        "energy": energy,
        "dissipation": dissipation,
        "reynolds": reynolds,
    }
    for name, values in series.items():
        ax.fill_between(
            time,
            values.min(axis=0),
            values.max(axis=0),
            color=COLORS[name],
            alpha=0.10,
            linewidth=0,
            zorder=1,
        )
    for curve in energy:
        ax.plot(time, curve, color=COLORS["energy"], alpha=0.13, linewidth=0.7)

    means = {
        "energy": energy.mean(axis=0),
        "dissipation": dissipation.mean(axis=0),
        "reynolds": reynolds.mean(axis=0),
    }
    labels = {
        "energy": r"$K/K_0$",
        "dissipation": r"$\varepsilon/\varepsilon_0$",
        "reynolds": r"$Re_\lambda/Re_{\lambda,0}$",
    }
    for name, values in means.items():
        ax.plot(
            time,
            values,
            color=COLORS[name],
            linewidth=2.15,
            solid_capstyle="round",
            linestyle={"energy": "-", "dissipation": "--", "reynolds": "-."}[name],
        )

    ax.axhline(1.0, color="#94a3b8", linewidth=0.85, linestyle=(0, (2, 2)))
    all_values = np.concatenate([energy.ravel(), dissipation.ravel(), reynolds.ravel()])
    padding = max(0.0012, 0.10 * (all_values.max() - all_values.min()))
    ax.set_ylim(all_values.min() - padding, all_values.max() + padding)
    ax.set_xlim(time[0], time[-1] + 0.004)
    ax.set_xlabel(r"Stored-window time $t-t_0$")
    ax.set_ylabel("Normalised statistic")
    ax.set_title("Stored-window dynamics", pad=9)
    ax.grid(axis="x", alpha=0.12)

    offsets = {"energy": -3, "dissipation": 4, "reynolds": -11}
    for name, values in means.items():
        ax.annotate(
            labels[name],
            xy=(time[-1], values[-1]),
            xytext=(-5, offsets[name]),
            textcoords="offset points",
            color=COLORS[name],
            fontsize=9.5,
            fontweight="bold",
            va="center",
            ha="right",
        )

    # (b) Per-trajectory diagnostics on fixed display reference scales.
    ax = axes[1]
    trajectory_ids = np.asarray([record["trajectory_id"] for record in records], dtype=int)
    divergence = np.asarray(
        [record["quality"]["max_divergence_rms"] for record in records], dtype=float
    )
    cfl = np.asarray([record["quality"]["max_cfl"] for record in records], dtype=float)

    cfl_reference = 0.5  # Also the generator's CFL acceptance threshold.
    divergence_reference = 1.5e-16  # Display scale for roundoff-level divergence.
    cfl_usage = cfl / cfl_reference
    divergence_usage = divergence / divergence_reference
    add_split_boundaries(ax)

    x = np.arange(len(trajectory_ids))
    width = 0.36
    ax.bar(
        x - width / 2,
        cfl_usage,
        width,
        color=COLORS["cfl"],
        alpha=0.88,
        label="CFL",
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        divergence_usage,
        width,
        color=COLORS["divergence"],
        alpha=0.88,
        label="Divergence",
        zorder=3,
    )
    ax.axhline(
        1.0,
        color=COLORS["limit"],
        linewidth=1.25,
        linestyle=(0, (3, 2)),
        zorder=4,
    )
    ax.set_xlim(-0.65, 14.65)
    ax.set_ylim(0, 1.10)
    ax.set_xticks(trajectory_ids[::2])
    ax.set_xlabel("Trajectory ID")
    ax.set_ylabel("Reference-scaled statistic")
    ax.set_title("Trajectory quality", pad=9)
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#cbd5e1",
        framealpha=0.9,
        fancybox=False,
        fontsize=10.0,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        borderaxespad=0,
    )
    ax.text(
        0.97,
        0.83,
        "Reference level = 1",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.2,
        color=COLORS["limit"],
    )
    place_panel_label_left_of_title(fig, axes[0], "(a)")
    place_panel_label_left_of_title(fig, axes[1], "(b)")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    plot()

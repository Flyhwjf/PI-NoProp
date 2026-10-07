"""Plot the local physics-loss weight sweep on its actual numerical coordinates.

Accuracy remains in percent, while the dimensionless weak residuals have their
own panel. Error bars show the aggregate's sample SD over seeds without clipping.
Only the lambda aggregate is read; decoder results belong in a separate table.
Default: input32/target16 residual-warm-start. Legacy: --protocol old16.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator
import numpy as np

from figure_protocol import FigureProtocol, V5, load_aggregate, parse_cli

ROOT = Path(__file__).resolve().parents[2]
LAMBDA_ARTIFACT = V5.aggregate_path("lambda_ablation")
# Provenance/compatibility constant; this figure does not read decoder results.
DECODER_ARTIFACT = V5.aggregate_path("decoder_ablation")
OUTPUT = ROOT / "paper/figures/fig_ablation.pdf"
FIGURE_WIDTH = 6.9
LINEAR_THRESHOLD = 0.001

CAPTION = (
    "Effect of the local physics-loss weight in the low-enstrophy group. "
    "(a) Five-class test accuracy in percent. (b) Contribution-normalised weak "
    "momentum and continuity residuals of the decoded fields. Points and error "
    "bars denote the mean and sample standard deviation over three seeds. "
    "Both panels use the actual numerical weights on a symmetric-logarithmic "
    "axis with a linear region up to 0.001, including zero. The physical "
    "condition, prototypes, decoder, data split, and local-update budget are "
    "held fixed across the sweep."
)


@dataclass(frozen=True)
class LambdaSweep:
    """Numerical plot data, in source units and increasing weight order."""
    weights: np.ndarray
    accuracy: np.ndarray
    accuracy_std: np.ndarray
    eta_ns: np.ndarray
    eta_ns_std: np.ndarray
    eta_div: np.ndarray
    eta_div_std: np.ndarray


def extract_lambda_data(artifact: dict) -> LambdaSweep:
    """Validate/extract declared weights without converting metric units.

    JSON spelling/order is not a plotting coordinate: "0" and "0.0" describe
    the same numerical weight, so records under both keys are ambiguous.
    """
    weights = np.asarray(artifact["protocol"]["lambdas"], dtype=float)
    if (weights.ndim != 1 or len(weights) < 3
            or not np.isfinite(weights).all() or (weights < 0).any()):
        raise ValueError("Expected at least three finite, non-negative lambda weights")
    if len(np.unique(weights)) != len(weights):
        raise ValueError("Lambda weights must be numerically distinct")
    weights = np.sort(weights)
    records = {}
    for key, record in artifact["results"].items():
        try:
            weight = float(key)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid lambda result key: {key!r}") from error
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Invalid lambda result key: {key!r}")
        if weight in records:
            raise ValueError(f"Duplicate numerical lambda result: {weight:g}")
        records[weight] = record
    if set(records) != set(weights):
        raise ValueError("Lambda result keys must match the declared numerical sweep")
    columns = []
    for key in ("accuracy", "eta_ns", "eta_div"):
        means = np.asarray([records[weight][key]["mean"] for weight in weights],
                           dtype=float)
        stds = np.asarray([records[weight][key]["std"] for weight in weights],
                          dtype=float)
        if (not np.isfinite(means).all() or not np.isfinite(stds).all()
                or (means < 0).any() or (stds < 0).any()):
            raise ValueError(f"{key}: expected finite non-negative means and SDs")
        if key == "accuracy" and (means > 100).any():
            raise ValueError("Accuracy means must be percentages in [0, 100]")
        columns.extend((means, stds))
    return LambdaSweep(weights, *columns)


def configure_style() -> None:
    """Times/STIX text remains readable at 5.4-inch manuscript placement."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11.5,
        "axes.labelsize": 12.0,
        "axes.titlesize": 12.0,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 11.5,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.grid": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def box_axes(axis: plt.Axes) -> None:
    """Draw all four spines consistently."""
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#465869")
        spine.set_linewidth(0.8)
    axis.tick_params(direction="out", length=3, width=0.7)


def place_panel_label_left_of_title(
    fig: plt.Figure, axis: plt.Axes, label: str
) -> None:
    """Place a marker immediately to the left of the centred title."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title_bbox = axis.title.get_window_extent(renderer=renderer)
    marker = axis.text(0, 0, label, transform=axis.transAxes, ha="right",
                       va="center", fontsize=12.0, fontweight="bold", clip_on=False)
    marker.set_in_layout(False)
    # The anchor is the RIGHT edge of a right-aligned marker.
    x_px = title_bbox.x0 - 5 * fig.dpi / 72
    y_px = title_bbox.y0 + 0.5 * title_bbox.height
    marker.set_position(axis.transAxes.inverted().transform((x_px, y_px)))


def lambda_tick(value: float) -> str:
    """Retain the scientific tick helper for callers using that style."""
    if value == 0:
        return "0"
    if value < 0 or not math.isfinite(value):
        raise ValueError(f"Invalid physics-loss weight: {value}")
    exponent = int(math.floor(math.log10(value)))
    coefficient = value / 10.0 ** exponent
    if math.isclose(coefficient, 1.0):
        return rf"$10^{{{exponent}}}$"
    return "$" + rf"{coefficient:g}\times10^{{{exponent}}}" + "$"


def errorbar_limits(means: np.ndarray, stds: np.ndarray,
                    *, minimum_top: float = 0.0) -> tuple[float, float]:
    """Include full mean +/- SD intervals, even beyond a metric's bounds."""
    lower = float(np.min(means - stds))
    upper = float(np.max(means + stds))
    return min(0.0, lower * 1.08), max(minimum_top, upper * 1.10, 0.01)


def make_figure(sweep: LambdaSweep) -> tuple[plt.Figure, np.ndarray]:
    """Build equal panels without reading sources or writing files."""
    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 3.45))
    fig.subplots_adjust(left=0.09, right=0.975, bottom=0.245, top=0.86, wspace=0.40)
    for axis in axes:
        box_axes(axis)
        axis.set_xscale("symlog", base=10, linthresh=LINEAR_THRESHOLD)
        axis.set_xticks(sweep.weights, [f"{weight:g}" for weight in sweep.weights])
        axis.tick_params(axis="x", labelrotation=40)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")
        axis.xaxis.set_minor_locator(NullLocator())
        padding = max(LINEAR_THRESHOLD * 0.2, float(sweep.weights[-1]) * 0.002)
        axis.set_xlim(float(sweep.weights[0]) - padding,
                      float(sweep.weights[-1]) * 1.30 + padding)
        axis.set_xlabel(r"Physics-loss weight $\lambda$")
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#D8DEE5", alpha=0.65, linewidth=0.6)
    common = {"linewidth": 1.55, "markersize": 4.5, "capsize": 3,
              "capthick": 0.8, "elinewidth": 0.9, "zorder": 3}
    axes[0].errorbar(sweep.weights, sweep.accuracy, yerr=sweep.accuracy_std,
                     color="#285C83", marker="o", markerfacecolor="white",
                     markeredgewidth=1.2, label="Test accuracy", **common)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_ylim(*errorbar_limits(sweep.accuracy, sweep.accuracy_std,
                                     minimum_top=100.0))
    axes[0].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0].set_title("Classification accuracy", pad=11)
    axes[0].legend(loc="lower left", frameon=True, fancybox=False, framealpha=0.92,
                   facecolor="white", edgecolor="#99A7B5", handlelength=1.7)
    axes[1].errorbar(sweep.weights, sweep.eta_ns, yerr=sweep.eta_ns_std,
                     color="#285C83", marker="s", markerfacecolor="#285C83",
                     label=r"Momentum $\eta_{\mathrm{NS}}$", **common)
    axes[1].errorbar(sweep.weights, sweep.eta_div, yerr=sweep.eta_div_std,
                     color="#6D8DA5", marker="^", markerfacecolor="white",
                     markeredgewidth=1.2, linestyle="--",
                     label=r"Continuity $\eta_{\mathrm{div}}$", **common)
    axes[1].set_ylabel("Normalised weak residual")
    axes[1].set_ylim(*errorbar_limits(
        np.concatenate((sweep.eta_ns, sweep.eta_div)),
        np.concatenate((sweep.eta_ns_std, sweep.eta_div_std)),
        minimum_top=1.10))
    axes[1].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[1].set_title("Decoded-field consistency", pad=11)
    axes[1].legend(loc="upper right", frameon=True, fancybox=False, framealpha=0.92,
                   facecolor="white", edgecolor="#99A7B5", handlelength=1.7)
    for index, axis in enumerate(axes):
        place_panel_label_left_of_title(fig, axis, f"({chr(97 + index)})")
    return fig, axes


def plot(protocol: FigureProtocol = V5, output: Path = OUTPUT) -> None:
    """Render only the selected protocol's lambda sweep, with no fallback."""
    sweep = extract_lambda_data(load_aggregate(protocol, "lambda_ablation"))
    fig, _ = make_figure(sweep)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        # A fixed canvas preserves 6.9-inch width and the intended font scale.
        fig.savefig(output, dpi=300, facecolor="white")
        preview = output.with_name(output.stem + "_preview.png")
        fig.savefig(preview, dpi=300, facecolor="white")
    finally:
        plt.close(fig)
    print(f"Wrote {output}")
    print(f"Preview: {preview}")


if __name__ == "__main__":
    plot(*parse_cli(OUTPUT, __doc__))

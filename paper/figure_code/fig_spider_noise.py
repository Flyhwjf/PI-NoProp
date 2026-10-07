"""SPIDER DNS-noise diagnostics, with three independent metric scales.

The source is the original DNS-field corruption sweep, independently of the
learning-cache protocol. Markers at the measured bootstrap percentages encode
the source's complete gate decision, including support identity and separation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
AGGREGATE = ROOT / "outputs/aggregate/full_ns_spider_noise.json"
OUTPUT = ROOT / "paper/figures/fig_spider_noise.pdf"
NS_SUPPORT = frozenset((
    "time_derivative", "convection", "pressure_gradient", "velocity_laplacian",
))


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10.0,
        "axes.labelsize": 10.2,
        "axes.titlesize": 10.2,
        "xtick.labelsize": 9.4,
        "ytick.labelsize": 9.8,
        "legend.fontsize": 9.6,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.grid": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": None,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_spider_artifact(source: Path = AGGREGATE) -> dict:
    """Read this exact DNS sweep; a missing file never selects another source."""
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"Missing SPIDER DNS-noise source: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def extract_spider_data(artifact: dict) -> dict:
    """Expose gate outcomes separately from support identity and repetition."""
    levels = np.asarray(artifact["levels_in_field_standard_deviations"], dtype=float)
    if (levels.ndim != 1 or len(levels) < 3 or not np.isfinite(levels).all()
            or np.any(levels < 0) or np.any(np.diff(levels) <= 0)):
        raise ValueError("DNS-noise levels must be finite, nonnegative and increasing")
    records = [artifact["levels"][str(float(level))] for level in levels]
    data = {"levels": levels, "positions": np.arange(len(levels)),
            "tick_labels": [f"{100 * level:g}" for level in levels]}
    for key in ("max_coefficient_relative_error", "bootstrap_support_fraction",
                "discovery_eta", "validation_eta", "test_eta"):
        values = np.asarray([record[key] for record in records], dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"Invalid SPIDER metric: {key}")
        if key == "bootstrap_support_fraction" and np.any(values > 1):
            raise ValueError("Bootstrap support fractions must lie in [0, 1]")
        if key != "bootstrap_support_fraction" and np.any(values == 0):
            raise ValueError(f"{key} must be positive for a logarithmic axis")
        data[key] = values
    # Percent conversion is only for display; residuals remain dimensionless.
    data["coefficient_error_percent"] = 100 * data["max_coefficient_relative_error"]
    data["bootstrap_percent"] = 100 * data["bootstrap_support_fraction"]
    flags = [record["validation_passed_under_noise_protocol"] for record in records]
    if not all(isinstance(value, bool) for value in flags):
        raise ValueError("SPIDER gate decisions must be explicit Boolean flags")
    data["passed"] = np.asarray(flags, dtype=bool)
    data["full_support"] = np.asarray([
        len(record["terms"]) == len(NS_SUPPORT)
        and frozenset(record["terms"]) == NS_SUPPORT for record in records
    ], dtype=bool)
    data["failure_reasons"] = tuple(tuple(record["failure_reasons"]) for record in records)
    if np.any(data["passed"] & ~data["full_support"]):
        raise ValueError("A passed corruption-sweep record must retain the full NS support")
    if any(passed and reasons for passed, reasons in
           zip(data["passed"], data["failure_reasons"])):
        raise ValueError("A passed corruption-sweep record has failure reasons")
    protocol = artifact["protocol"]
    data["coefficient_gate_percent"] = 100 * float(protocol["max_coefficient_relative_error"])
    data["bootstrap_gate_percent"] = 100 * float(protocol["min_bootstrap_support"])
    data["validation_gate"] = float(protocol["max_validation_eta"])
    data["test_gate"] = float(protocol["max_test_eta"])
    return data


def figure_caption() -> str:
    return (
        "SPIDER equation discovery under Gaussian DNS-field noise. "
        "(a) Maximum relative coefficient error. "
        "(b) Bootstrap repetition frequency of the selected support. "
        "(c) Contribution-normalised discovery, validation and held-out test "
        "weak residuals. Dashed lines mark the corruption-sweep gates: "
        "200% coefficient error, 50% bootstrap repetition and 0.25 residual. "
        "Markers at the actual bootstrap percentages in (b) encode the complete "
        "corruption-sweep decision (filled circles: pass; crosses: fail), including full "
        "momentum support and support separation. Bootstrap repetition "
        "measures the selected support at each level, including selections "
        "with additional terms. Tested noise levels are equally spaced "
        "categorical positions labelled as percentages of field standard "
        "deviation; panels (a) and (c) use logarithmic vertical scales."
    )


def build_figure(artifact: dict) -> plt.Figure:
    data = extract_spider_data(artifact)
    configure_style()
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 3.65))
    fig.subplots_adjust(left=0.083, right=0.985, bottom=0.27, top=0.87, wspace=0.48)
    x = data["positions"]
    dark, blue, grey = "#244D70", "#4D87AD", "#7B8793"
    ax = axes[0]
    ax.plot(x, data["coefficient_error_percent"], color=dark,
            marker="o", markersize=4.0, linewidth=1.5, zorder=3)
    ax.set_yscale("log")
    minimum = min(float(data["coefficient_error_percent"].min()) * 0.4, 0.01)
    maximum = max(float(data["coefficient_error_percent"].max()) * 2.5,
                  data["coefficient_gate_percent"] * 3)
    ax.set_ylim(minimum, maximum)
    ax.set_ylabel("Maximum coefficient error (%)")
    gate = data["coefficient_gate_percent"]
    ax.axhline(gate, color=grey, linestyle="--", linewidth=0.9, zorder=1)
    ax.text(0.045, gate * 1.23, f"{gate:g}% gate",
            transform=ax.get_yaxis_transform(), fontsize=9.6,
            color="#596773", ha="left", va="bottom")
    ax.set_title("(a)  Coefficient error", loc="left", pad=9)

    ax = axes[1]
    ax.plot(x, data["bootstrap_percent"], color=blue,
            marker=None, linewidth=1.5, zorder=3)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Selected-support repetition (%)")
    gate = data["bootstrap_gate_percent"]
    ax.axhline(gate, color=grey, linestyle="--", linewidth=0.9, zorder=1)
    ax.text(0.045, gate - 3.0, f"{gate:g}% gate",
            transform=ax.get_yaxis_transform(), fontsize=9.6,
            color="#596773", ha="left", va="top")
    # Measured percentages set the coordinates; marker shapes encode the
    # independent, joint corruption-sweep decision at those same coordinates.
    for passed, marker in ((True, "o"), (False, "x")):
        chosen = data["passed"] == passed
        ax.scatter(x[chosen], data["bootstrap_percent"][chosen], marker=marker,
                   s=23, color=dark if passed else grey, linewidths=1.0, zorder=4)
    ax.legend(handles=[
        Line2D([], [], color=dark, marker="o", linestyle="none",
               markersize=4.0, label="Pass"),
        Line2D([], [], color=grey, marker="x", linestyle="none",
               markersize=4.0, label="Fail"),
    ], title="Protocol status", title_fontsize=9.6, loc="lower right",
        ncol=2, fontsize=9.6, frameon=True, fancybox=False,
        framealpha=0.85, edgecolor="#93A1AE", facecolor="white",
        handlelength=0.8, handletextpad=0.25, columnspacing=0.8, borderpad=0.3)
    ax.set_title("(b)  Bootstrap stability", loc="left", pad=9)

    ax = axes[2]
    for key, label, colour, marker, linestyle, fill in (
        ("discovery_eta", "Discovery", grey, "D", ":", "white"),
        ("validation_eta", "Validation", blue, "o", "--", "white"),
        ("test_eta", "Held-out test", dark, "s", "-", dark),
    ):
        ax.plot(x, data[key], color=colour, marker=marker, linestyle=linestyle,
                markerfacecolor=fill, markeredgewidth=0.8,
                markersize=3.8, linewidth=1.3, label=label, zorder=3)
    ax.set_yscale("log")
    lower = min(float(data[key].min()) for key in
                ("discovery_eta", "validation_eta", "test_eta")) * 0.5
    upper = max(float(data[key].max()) for key in
                ("discovery_eta", "validation_eta", "test_eta")) * 2.5
    ax.set_ylim(lower, max(upper, data["validation_gate"] * 4))
    ax.set_ylabel(r"Normalised weak residual $\eta$")
    # The current validation and test gates coincide. Keep distinct values if
    # a future source declares separate gates, without silently merging them.
    gates = [(data["validation_gate"], "Validation gate"),
             (data["test_gate"], "Test gate")]
    if data["validation_gate"] == data["test_gate"]:
        gates = [(data["validation_gate"], "gate")]
    for gate, name in gates:
        ax.axhline(gate, color=grey, linestyle="--", linewidth=0.9, zorder=1)
        ax.text(0.045, gate * 1.25, f"{gate:g} {name.lower()}",
                transform=ax.get_yaxis_transform(), fontsize=9.6,
                color="#596773", ha="left", va="bottom")
    ax.legend(loc="lower right", frameon=True, fancybox=False, framealpha=0.85,
              edgecolor="#93A1AE", facecolor="white", borderpad=0.3,
              handlelength=1.25, handletextpad=0.4, labelspacing=0.3)
    ax.set_title("(c)  Weak residual", loc="left", pad=9)

    for axis in axes:
        axis.set_xlim(-0.4, len(x) - 0.6)
        axis.set_xticks(x, data["tick_labels"], rotation=45, ha="right")
        axis.set_xlabel("DNS noise / field SD (%)", labelpad=5)
        axis.tick_params(length=3, width=0.7)
        axis.grid(axis="y", which="major", color="#E1E6EA", linewidth=0.55, zorder=0)
        axis.minorticks_off()
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("#566575")
            spine.set_linewidth(0.75)
    return fig


def plot() -> None:
    artifact = load_spider_artifact()
    fig = build_figure(artifact)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    preview = OUTPUT.with_name(f"{OUTPUT.stem}_preview.png")
    try:
        fig.savefig(OUTPUT, format="pdf", dpi=300, facecolor="white",
                    metadata={"Title": "SPIDER DNS-noise diagnostics",
                              "Subject": figure_caption()})
        fig.savefig(preview, format="png", dpi=300, facecolor="white")
    finally:
        plt.close(fig)
    print(f"Source: {AGGREGATE}\nWrote {OUTPUT}\nPreview: {preview}")


if __name__ == "__main__":
    plot()

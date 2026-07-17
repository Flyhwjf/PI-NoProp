"""Generate figures that correspond to the current validated experiment protocol."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "paper" / "figures" / "legacy"
RESULTS = ROOT / "outputs" / "aggregate" / "discovery_results.json"

COLORS = {
    "blue": "#2F6690", "cyan": "#3A8DAD", "green": "#3A9365",
    "orange": "#D97732", "red": "#C44E52", "grey": "#667085",
    "light": "#F3F6F9", "dark": "#1F2937",
}
METHODS = ("none", "analytic", "spider")
METHOD_LABELS = {"none": "Local NoProp", "analytic": "Analytic prior",
                 "spider": "Discovered relation"}
METHOD_COLORS = {"none": COLORS["grey"], "analytic": COLORS["orange"],
                 "spider": COLORS["green"]}


def setup_style() -> None:
    plt.rcParams.update({
        "font.size": 9.5, "axes.titlesize": 11, "axes.labelsize": 9.5,
        "legend.fontsize": 8.5, "figure.dpi": 120, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.7,
    })


def save(fig: plt.Figure, name: str) -> None:
    path = FIGURES / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path.relative_to(ROOT))


def box(ax, xy, width, height, label, color, fontsize=9.5):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=color, edgecolor="white", linewidth=1.4)
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, label,
            ha="center", va="center", color="white", fontsize=fontsize,
            fontweight="semibold", linespacing=1.25)
    return patch


def arrow(ax, start, end, color=COLORS["dark"], style="-"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.5,
        color=color, linestyle=style, shrinkA=3, shrinkB=3))


def framework_overview() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 3.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    xs = [0.025, 0.27, 0.515, 0.76]
    labels = [
        "HIT snapshots\n60 discovery\n20 validation",
        "SPIDER\nsparse weak-form\ndiscovery",
        "Validated artifact\nterms + coefficients\nprovenance",
        "Local NoProp\ndiscovered residual\nin each block",
    ]
    colors = [COLORS["blue"], COLORS["cyan"], COLORS["orange"], COLORS["green"]]
    for x, label, color in zip(xs, labels, colors):
        box(ax, (x, 0.43), 0.20, 0.28, label, color, fontsize=7.8)
    for x in xs[:-1]:
        arrow(ax, (x + 0.202, 0.57), (x + 0.243, 0.57))
    ax.text(0.125, 0.27, r"$64^3$ velocity and pressure",
            ha="center", color=COLORS["grey"])
    ax.text(0.37, 0.27,
            r"$\nabla^2p+0.293\,\nabla\!\cdot[(u\!\cdot\!\nabla)u]=0$",
            ha="center", color=COLORS["grey"])
    ax.text(0.615, 0.27, "held-out residual 0.286\n100% bootstrap support",
            ha="center", color=COLORS["grey"])
    ax.text(0.86, 0.27,
            "strictly isolated optimizers\nphysical consistency evaluation",
            ha="center", color=COLORS["grey"])
    ax.text(0.5, 0.88, "Physics-informed NoProp pipeline",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=COLORS["dark"])
    save(fig, "fig_framework_overview.png")


def local_training_diagram() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, (0.025, 0.61), 0.15, 0.18, "Frozen encoder\ncondition $c(x)$",
        COLORS["blue"])
    box(ax, (0.025, 0.22), 0.15, 0.18, "Noisy target\n$z_t$",
        COLORS["cyan"])
    y_positions = [0.70, 0.50, 0.30]
    labels = ["Block 1", r"Sampled block $J$", "Block $T$"]
    for y, label in zip(y_positions, labels):
        active = "Sampled" in label
        box(ax, (0.29, y - 0.075), 0.17, 0.15, label,
            COLORS["green"] if active else COLORS["grey"])
        arrow(ax, (0.175, 0.70), (0.29, y), COLORS["blue"],
              ":" if not active else "-")
        arrow(ax, (0.175, 0.31), (0.29, y), COLORS["cyan"],
              ":" if not active else "-")
    ax.text(0.375, 0.12, "one independent optimizer per block",
            ha="center", color=COLORS["grey"])
    box(ax, (0.57, 0.58), 0.17, 0.20,
        "Frozen decoder\n$z_J\\mapsto(\\tilde u,\\tilde p)$", COLORS["orange"])
    box(ax, (0.57, 0.24), 0.17, 0.20,
        "Local objective\n$T(\\mathcal{L}_{diff}+\\lambda\\mathcal{L}_{phys})$",
        COLORS["red"])
    arrow(ax, (0.46, 0.50), (0.57, 0.68), COLORS["green"])
    arrow(ax, (0.655, 0.58), (0.655, 0.44), COLORS["orange"])
    arrow(ax, (0.57, 0.34), (0.46, 0.50), COLORS["red"])
    box(ax, (0.81, 0.58), 0.16, 0.20, "Detached $z_T$\nclassifier training",
        COLORS["blue"])
    arrow(ax, (0.74, 0.68), (0.81, 0.68), COLORS["dark"], ":")
    ax.text(0.81, 0.35, "gradient isolation", ha="left", color=COLORS["red"],
            fontweight="bold")
    ax.plot([0.79, 0.97], [0.31, 0.31], color=COLORS["red"], linewidth=2)
    ax.text(0.88, 0.235, "no classifier-to-block or\ncross-block gradient",
            ha="center", va="top", color=COLORS["grey"])
    ax.text(0.5, 0.93, "Strictly local block update",
            ha="center", fontsize=13, fontweight="bold", color=COLORS["dark"])
    save(fig, "fig_local_training.png")


def main_results(results: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.55),
                             constrained_layout=True)
    regions = ("centre", "edge")
    x = np.arange(len(regions))
    width = 0.24
    metrics = [
        ("accuracy_mean", "accuracy_std", "Test accuracy (%)", "Classification"),
        ("eta_div_mean", None, r"Continuity residual $\eta_{div}$", "Continuity"),
        ("eta_pp_discovered_mean", None, r"Discovered residual $\eta_{SP}$",
         "Discovered relation"),
    ]
    for ax, (metric, error, ylabel, title) in zip(axes, metrics):
        for idx, method in enumerate(METHODS):
            vals = [results["main"][r][method][metric] for r in regions]
            errs = ([results["main"][r][method][error] for r in regions]
                    if error else None)
            ax.bar(x + (idx - 1) * width, vals, width, yerr=errs, capsize=3,
                   label=METHOD_LABELS[method], color=METHOD_COLORS[method],
                   edgecolor="white", linewidth=0.6)
        ax.set_xticks(x, ["Centre", "Edge"])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, loc="upper right")
    save(fig, "fig_main_results.png")


def load_validation_curves(region: str, method: str) -> np.ndarray:
    token = {"none": "none_none", "analytic": "analytic_divpp",
             "spider": "spider_divpp"}[method]
    curves = []
    for seed in (0, 1, 2):
        run = (ROOT / "outputs" / "runs" /
               f"local_fast_{token}_{region}_lambda0p01_seed{seed}_steps100" /
               "history.npz")
        history = np.load(run, allow_pickle=True)["classifier"]
        curves.append([entry["val"]["accuracy"] for entry in history])
    return np.asarray(curves, dtype=float)


def training_dynamics() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.7, 3.55), sharey=True,
                             constrained_layout=True)
    for ax, region in zip(axes, ("centre", "edge")):
        for method in METHODS:
            values = load_validation_curves(region, method)
            epochs = np.arange(1, values.shape[1] + 1)
            mean = values.mean(axis=0)
            std = values.std(axis=0, ddof=1)
            ax.plot(epochs, mean, color=METHOD_COLORS[method], linewidth=1.8,
                    label=METHOD_LABELS[method])
            ax.fill_between(epochs, mean - std, mean + std,
                            color=METHOD_COLORS[method], alpha=0.13, linewidth=0)
        ax.set_title(f"{region.capitalize()} validation split")
        ax.set_xlabel("Classifier epoch")
        ax.set_ylim(0, 55)
        ax.set_ylabel("Accuracy (%)")
    axes[0].legend(frameon=False, loc="upper left")
    save(fig, "fig_current_training_dynamics.png")


def efficiency(results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5),
                             constrained_layout=True)
    x = np.arange(len(METHODS))
    width = 0.34
    for ridx, region in enumerate(("centre", "edge")):
        shift = (ridx - 0.5) * width
        times = [results["main"][region][m]["train_seconds_mean"] for m in METHODS]
        memory = [results["main"][region][m]["peak_memory_mb_mean"] for m in METHODS]
        color = COLORS["blue"] if region == "centre" else COLORS["cyan"]
        axes[0].bar(x + shift, times, width, color=color, label=region.capitalize())
        axes[1].bar(x + shift, memory, width, color=color, label=region.capitalize())
    for ax in axes:
        ax.set_xticks(x, ["None", "Analytic", "SPIDER"])
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Complete block phase (s)")
    axes[0].set_title("Training time")
    axes[1].set_ylabel("Peak GPU memory (MB)")
    axes[1].set_title("GPU memory")
    axes[0].legend(frameon=False)
    save(fig, "fig_efficiency.png")


def main() -> None:
    setup_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    framework_overview()
    local_training_diagram()
    main_results(results)
    training_dynamics()
    efficiency(results)


if __name__ == "__main__":
    main()

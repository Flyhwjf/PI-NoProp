"""Vector schematic of the current input32/target16 PI-NoProp implementation.

This anatomy is specific to v5. All arrows are forward computations except the
single dashed selected-block update. Frozen decoder parameters still permit
differentiation with respect to its latent input. Future fields are used during
shared preparation, whereas inference consumes a first-frame condition only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as DrawingPath


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "paper/figures/fig_framework.pdf"
PROTOCOL = "full_ns_v5_input32_target16_residual_warmstart"
SOURCE_CODE = (
    "src/noprop/model.py",
    "src/training/pretrain.py",
    "src/training/local_trainer.py",
    "src/decoder.py",
    "src/physics/temporal_ns_loss.py",
    "src/noprop/classifier.py",
)
CAPTION = (
    "Data and gradient flow in PI-NoProp. (a) A first-frame 32-cubed context "
    "and its centred 16-cubed core are encoded with a gated spatial pathway "
    "and an equation-derived physical rate. SPIDER estimates the momentum "
    "coefficients from discovery trajectories. Centre-condition pretraining, "
    "class-prototype construction, temporal-decoder pretraining, and context "
    "adaptation precede freezing the shared modules and caching the "
    "128-dimensional conditions. The decoder is supervised with nine-frame "
    "centred velocity-pressure windows during preparation. (b) A selected "
    "block denoises an analytically sampled label-noise marginal. Its updated "
    "latent is decoded into a nine-frame field window, and the weak momentum "
    "and continuity losses join the prototype-denoising objective. Gradients "
    "pass through the frozen decoder to this block only. (c) Inference uses "
    "the first-frame condition, ten sequential updates from Gaussian noise, "
    "and cosine similarity to the frozen five-class prototypes. Solid arrows "
    "denote forward computation; the dashed arrow denotes the local update."
)

INK = "#263D50"
BORDER = "#9DAAB6"
BLUE = "#285C83"
PALE_BLUE = "#EAF1F7"
FROZEN = "#F4F6F8"


def configure_style() -> None:
    """11.5+ pt native text gives 9+ pt at the 5.4-inch manuscript width."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def box(ax, x, y, width, height, text, *, active=False, physical=False):
    """An explicitly bounded node, with readable centred multiline text."""
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.025,rounding_size=0.07",
        facecolor=PALE_BLUE if active or physical else FROZEN,
        edgecolor=BLUE if active or physical else BORDER,
        linewidth=1.4 if active else 0.85,
        zorder=3,
    )
    ax.add_patch(patch)
    label = ax.text(x + width / 2, y + height / 2, text,
                    ha="center", va="center", color=INK,
                    fontsize=11.5, linespacing=1.18, zorder=4)
    # Store bounds for read-only geometry checks after rendering.
    label._framework_node_bounds = (x, y, width, height)
    return patch, label


def arrow(ax, points, *, gradient=False):
    """Use a solid forward path or the single dashed local-gradient path."""
    codes = [DrawingPath.MOVETO] + [DrawingPath.LINETO] * (len(points) - 1)
    path = DrawingPath(points, codes)
    patch = FancyArrowPatch(
        path=path, arrowstyle="-|>", mutation_scale=10.5,
        color=BLUE if gradient else "#647C8E",
        linewidth=1.4 if gradient else 1.0,
        linestyle=(0, (4, 2.5)) if gradient else "solid",
        zorder=2,
    )
    ax.add_patch(patch)
    return patch


def strip(ax, y, height, heading):
    """A lightly outlined stage, rather than a coloured background region."""
    patch = FancyBboxPatch(
        (0.15, y), 11.70, height,
        boxstyle="round,pad=0.02,rounding_size=0.09",
        facecolor="white", edgecolor="#C4CDD5", linewidth=0.8,
        zorder=0,
    )
    ax.add_patch(patch)
    ax.text(0.40, y + height - 0.36, heading, fontsize=13.0,
            ha="left", va="center", color=INK, fontweight="bold")


def make_figure():
    """Draw the preparation/training/inference anatomy without model execution."""
    configure_style()
    fig, ax = plt.subplots(figsize=(6.9, 7.35))
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.995)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 13.7)
    ax.axis("off")

    # (a) Shared preparation and the two first-frame conditioning paths.
    strip(ax, 8.10, 5.28, "(a) Shared preparation")
    box(ax, 0.40, 11.25, 2.90, 1.05,
        "$64^3$ DNS\nDisjoint trajectories")
    box(ax, 4.20, 11.25, 3.10, 1.05,
        "First-frame input\n$4\\times32^3$; centred $16^3$")
    box(ax, 8.15, 11.25, 3.45, 1.05,
        "Weak-form SPIDER\nAccepted coefficients $\\mathbf{c}$", physical=True)
    arrow(ax, [(3.30, 11.775), (4.20, 11.775)])
    arrow(ax, [(1.85, 12.30), (1.85, 12.60),
               (9.875, 12.60), (9.875, 12.30)])
    ax.text(6.80, 12.76, "Discovery split", fontsize=11.5,
            color=BLUE, ha="center", va="center")

    box(ax, 0.40, 9.60, 3.75, 1.22,
        "Centre + gated context\n$16^3$ centre / $32^3$ context")
    box(ax, 4.80, 9.60, 2.45, 1.22,
        "Condition $\\mathbf{h}$\n128 dimensions", physical=True)
    box(ax, 8.15, 9.60, 3.45, 1.22,
        "First-frame physical rate\nConvective / pressure /\nviscous energy rates ($16^3$)", physical=True)
    arrow(ax, [(5.75, 11.25), (5.75, 11.04),
               (2.275, 11.04), (2.275, 10.82)])
    arrow(ax, [(7.30, 11.775), (7.72, 11.775),
               (7.72, 10.60), (8.15, 10.60)])
    arrow(ax, [(9.875, 11.25), (9.875, 10.82)])
    arrow(ax, [(4.15, 10.21), (4.80, 10.21)])
    arrow(ax, [(8.15, 10.21), (7.25, 10.21)])

    # The preparation chronology is separate from the frozen anatomy above.
    ax.text(6.0, 9.20,
            r"Prepare: centre condition $\rightarrow$ prototypes $e_y$ / temporal decoder",
            ha="center", va="center", color=INK, fontsize=11.5)
    ax.text(6.0, 8.83,
            r"Adapt gated $32^3$ context; freeze shared modules and cache $\mathbf{h}$.",
            ha="center", va="center", color=INK, fontsize=11.5)
    ax.text(6.0, 8.46,
            r"Decoder supervision: $9\times4\times16^3$ fields from the training split.",
            ha="center", va="center", color=INK, fontsize=11.5)

    # (b) Each update constructs the label-noise marginal directly. No other
    # NoProp block is evaluated in this local-training diagram.
    strip(ax, 2.60, 5.27, "(b) Local training: one selected block")
    ax.text(4.625, 7.12, r"Cached condition $\mathbf{h}$", fontsize=11.5,
            ha="center", va="center", color=BLUE)
    arrow(ax, [(4.625, 6.99), (4.625, 6.90)])
    box(ax, 0.40, 5.70, 2.35, 1.20,
        "$e_y$ + Gaussian noise\nAnalytic $z_t^{\\mathrm{in}}$")
    box(ax, 3.30, 5.70, 2.65, 1.20,
        "Selected block $f_t$\nPrediction $\\hat e_t$", active=True)
    box(ax, 6.50, 5.70, 2.35, 1.20,
        "Latent update\n$z_{t+1}=a_t\\hat e_t$\n$+\\,b_tz_t^{\\mathrm{in}}$")
    box(ax, 9.40, 5.70, 2.25, 1.20,
        "Frozen decoder $D_\\phi$\n$9\\times4\\times16^3$\nVelocity / pressure")
    arrow(ax, [(2.75, 6.30), (3.30, 6.30)])
    arrow(ax, [(5.95, 6.30), (6.50, 6.30)])
    arrow(ax, [(8.85, 6.30), (9.40, 6.30)])

    box(ax, 2.20, 4.00, 3.80, 1.07,
        "Prototype-denoising loss\n$L_{\\mathrm{diff}}=0.5\\,\\mathrm{MSE}(\\hat e_t,e_y)$")
    box(ax, 7.00, 4.00, 4.65, 1.07,
        "Weak physical loss\n$L_{\\mathrm{phys}}=\\langle\\eta_{\\mathrm{NS}}^2\\rangle"
        "+0.25\\langle\\eta_{\\mathrm{div}}^2\\rangle$", physical=True)
    arrow(ax, [(4.625, 5.70), (4.625, 5.07)])
    arrow(ax, [(10.525, 5.70), (10.525, 5.07)])
    box(ax, 2.00, 2.83, 8.45, 0.90,
        "$L_t=10\\,(L_{\\mathrm{diff}}+0.1\\,L_{\\mathrm{phys}})$\n"
        "Gradient to $z_{t+1}$ through frozen $D_\\phi$; update $f_t$ only", active=True)
    arrow(ax, [(4.10, 4.00), (4.10, 3.73)])
    arrow(ax, [(9.325, 4.00), (9.325, 3.73)])
    arrow(ax, [(2.00, 3.28), (0.38, 3.28), (0.38, 5.30),
               (3.78, 5.30), (3.78, 5.70)], gradient=True)
    ax.text(1.45, 5.46, "Local gradient", fontsize=11.5,
            color=BLUE, ha="center", va="center")

    # (c) Label-free first-frame conditioning is the only data input at inference.
    strip(ax, 0.18, 2.20, "(c) Inference from a first frame")
    box(ax, 0.40, 0.73, 2.50, 1.05,
        "Gaussian $z_0$\n128 dimensions")
    box(ax, 3.45, 0.73, 5.10, 1.05,
        "First-frame condition $\\mathbf{h}$\n"
        "$f_0\\rightarrow\\cdots\\rightarrow f_9\\rightarrow z_{10}$")
    box(ax, 9.20, 0.73, 2.40, 1.05,
        "Cosine readout\nFrozen $e_0,\\ldots,e_4$\nFive-class output")
    arrow(ax, [(2.90, 1.255), (3.45, 1.255)])
    arrow(ax, [(8.55, 1.255), (9.20, 1.255)])
    ax.text(6.0, 0.43,
            "Solid: forward computation       Dashed: selected-block gradient",
            fontsize=11.5, ha="center", va="center", color=INK)
    return fig, ax


def node_text_overflows(fig, ax):
    """Return labels extending beyond their declared boxes (read-only check)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    overflows = []
    for label in ax.texts:
        bounds = getattr(label, "_framework_node_bounds", None)
        if bounds is None:
            continue
        x, y, width, height = bounds
        (left, bottom), (right, top) = ax.transData.transform(
            [(x, y), (x + width, y + height)])
        extent = label.get_window_extent(renderer)
        if (extent.x0 < left or extent.x1 > right
                or extent.y0 < bottom or extent.y1 > top):
            overflows.append(label.get_text())
    return overflows


def plot(output: Path = OUTPUT) -> None:
    """Write the v5-only schematic as a vector PDF and a 300-dpi PNG preview."""
    fig, ax = make_figure()
    try:
        overflows = node_text_overflows(fig, ax)
        if overflows:
            raise RuntimeError("Framework text exceeds node bounds: " + repr(overflows))
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, facecolor="white")
        preview = output.with_name(output.stem + "_preview.png")
        fig.savefig(preview, dpi=300, facecolor="white")
    finally:
        plt.close(fig)
    print(f"Wrote {output}")
    print(f"Preview: {preview}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    plot(parser.parse_args().output)

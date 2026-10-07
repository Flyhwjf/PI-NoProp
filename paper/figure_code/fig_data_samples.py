"""Generate the representative decaying-HIT sample-field figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from figure_protocol import FigureProtocol, V5, parse_cli, read_json


ROOT = Path(__file__).resolve().parents[2]
DATA = V5.cache_path
RAW_FRAME = ROOT / "data/generated_hit_ns/trajectory_000/frame_000.npz"
OUTPUT = ROOT / "paper/figures/fig_data_samples.pdf"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Times"],
            "mathtext.fontset": "stix",
            "font.size": 11.0,
            "axes.titlesize": 12.0,
            "axes.labelsize": 11.0,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "figure.dpi": 160,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def show_full_frame(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#334155")
        spine.set_linewidth(0.8)


def plot(protocol: FigureProtocol = V5, output: Path = OUTPUT) -> None:
    configure_style()
    data = protocol.cache_path
    metadata = read_json(data / "metadata.json")
    input_size = metadata.get("input_spatial_size", metadata.get("spatial_size"))
    target_size = metadata.get("target_spatial_size", metadata.get("spatial_size"))
    if (input_size, target_size) != (protocol.input_size, protocol.target_size):
        raise ValueError(f"Cache does not match {protocol.name}: {data}")
    fields = np.load(data / "fields.npy", mmap_mode="r")
    regions = np.load(data / "regions.npy")
    if fields.shape[2:] != (protocol.input_size,) * 3 or len(fields) != len(regions):
        raise ValueError(f"Invalid input fields/regions in {data}: {fields.shape}")
    print(f"Source: {data / 'fields.npy'} (raw input u_x, not the target sequence)")

    with np.load(RAW_FRAME) as raw:
        full_dns_field = np.asarray(raw["velocity"][0])
    print(f"Source: {RAW_FRAME} (unchanged full DNS)")

    selected_fields = [full_dns_field]
    for region in (0, 1):
        indices = np.flatnonzero(regions == region)
        if not len(indices):
            raise ValueError(f"No samples for enstrophy group {region} in {data}")
        index = int(indices[0])
        selected_fields.append(np.asarray(fields[index, 0]))

    limit = max(np.max(np.abs(value)) for value in selected_fields)

    fig = plt.figure(figsize=(6.9, 6.0))
    panel_width = 0.22
    panel_height = panel_width * fig.get_figwidth() / fig.get_figheight()
    row_gap = 0.035
    column_gap = row_gap * fig.get_figheight() / fig.get_figwidth()
    left = 0.10
    bottom = 0.065
    x_positions = [left + column * (panel_width + column_gap) for column in range(3)]
    y_positions = [
        bottom + 2 * (panel_height + row_gap),
        bottom + panel_height + row_gap,
        bottom,
    ]

    axes = np.empty((3, 3), dtype=object)
    for row in range(3):
        for column in range(3):
            axes[row, column] = fig.add_axes(
                [x_positions[column], y_positions[row], panel_width, panel_height]
            )

    row_labels = (
        r"Full DNS ($64^3$)",
        rf"Low enstrophy (${protocol.input_size}^3$)",
        rf"High enstrophy (${protocol.input_size}^3$)",
    )
    images = []
    for row, field in enumerate(selected_fields):
        slices = (
            field[field.shape[0] // 2],
            field[:, field.shape[1] // 2],
            field[:, :, field.shape[2] // 2],
        )
        for column, value in enumerate(slices):
            ax = axes[row, column]
            image = ax.imshow(
                value.T,
                origin="lower",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                interpolation="nearest",
                aspect="equal",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            show_full_frame(ax)
            if row > 0 and protocol.input_size > protocol.target_size:
                edge = (protocol.input_size - protocol.target_size) / 2 - 0.5
                ax.add_patch(Rectangle(
                    (edge, edge), protocol.target_size, protocol.target_size,
                    fill=False, edgecolor="#334155", linewidth=1.0,
                    linestyle=(0, (3, 2)),
                ))
            ax.set_title(("x", "y", "z")[column] + "-normal", pad=4)
            if column == 0:
                ax.set_ylabel(row_labels[row], labelpad=10)
        images.append(image)

    colorbar_x = x_positions[-1] + panel_width + 0.025
    for row in range(3):
        colorbar_axis = fig.add_axes(
            [colorbar_x, y_positions[row], 0.022, panel_height]
        )
        colorbar = fig.colorbar(images[row], cax=colorbar_axis)
        colorbar.ax.set_title(r"$u_x$", fontsize=10.5, pad=4)
        colorbar.ax.tick_params(labelsize=9.5)
        colorbar.outline.set_visible(True)
        colorbar.outline.set_linewidth(0.8)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    plot(*parse_cli(OUTPUT, __doc__))

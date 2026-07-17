"""Comprehensive plotting utilities for PI-NoProp training and evaluation.

All figure-generation functions used by the pi_noprop_plots notebook
live here so the notebook stays clean execution-only.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MultipleLocator
import torch
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def set_global_style():
    """Set matplotlib style for uniform-looking figures."""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 120
    plt.rcParams['savefig.dpi'] = 150
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['mathtext.fontset'] = 'stix'


# ─────────────────────────────────────────────────────────────
#  1.  Training curves (multi-panel)
# ─────────────────────────────────────────────────────────────

def plot_training_curves(history, label=None, save_path=None):
    """Fig C: 1×2 layout — left panel loss + accuracy (twin-y), right panel loss components.

    Args:
        history: dict with keys train_loss, val_loss, val_acc, cls_loss, diff_loss, phys_loss
        label: optional run label (unused, kept for signature compatibility)
    """
    epochs = np.arange(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ── Left: loss (left-y, black) + accuracy (right-y, green) ──
    ax1 = axes[0]
    ax1.plot(epochs, history['train_loss'], '-', color='black', lw=2, label='Train Loss')
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], '--', color='black', lw=1.5, label='Val Loss')
    ax1.set_xlabel('Epoch', fontsize=20)
    ax1.set_ylabel('Loss', fontsize=20)
    ax1.tick_params(labelsize=20)
    ax1.set_yscale('log')
    ax1.set_ylim(1, 100)
    ax1.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax1.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax1.grid(True, alpha=0.3, which='both')

    ax2 = ax1.twinx()
    ax2.plot(epochs, history['val_acc'], '-', color='#1b7837', lw=2, label='Accuracy')
    ax2.set_ylabel('Accuracy (%)', fontsize=20)
    ax2.tick_params(labelsize=20)
    ax2.yaxis.set_major_locator(MultipleLocator(20))
    ax2.set_ylim(0, 100)

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc='upper right', fontsize=18)
    ax1.set_title('Training Loss & Accuracy', fontsize=22)

    # ── Right: loss components (colour-coded) ──
    ax = axes[1]
    component_style = [
        ('cls_loss', 'Classification', '#e66101'),
        ('diff_loss', 'Diffusion', '#5e3c99'),
        ('phys_loss', 'Physics', '#994d00'),
    ]
    for key, label, color in component_style:
        if key in history and len(history[key]) > 0:
            ls = '--' if key == 'phys_loss' else '-'
            ax.plot(epochs, history[key], ls, color=color, lw=1.5, label=label)
    ax.set_xlabel('Epoch', fontsize=20)
    ax.set_ylabel('Loss', fontsize=20)
    ax.tick_params(labelsize=20)
    ax.set_title('Loss Components', fontsize=22)
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 10)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────
#  2.  Data samples (Centre vs Edge velocity slices)
# ─────────────────────────────────────────────────────────────

def plot_data_samples(data_dir, n_slices=3, save_path=None):
    """Centre vs edge streamwise-velocity slices from generated HIT data.

    The stored velocity layout is (component, time, x, y, z). This function
    plots three orthogonal slices of u_x at frame zero and uses one shared
    colour range so that the two gradient groups are directly comparable.

    Args:
        data_dir: root data directory containing 'centre/' and 'edge/'
        n_slices: number of slices to show (default 3)
    """
    data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
    d_centre = np.load(data_dir / 'centre' / 'sub_0000.npz')
    d_edge = np.load(data_dir / 'edge' / 'sub_0000.npz')

    velocity_centre = d_centre['velocity']
    velocity_edge = d_edge['velocity']
    if velocity_centre.ndim != 5 or velocity_centre.shape[0] != 3:
        raise ValueError(
            f'Expected velocity layout (3, time, x, y, z), got '
            f'{velocity_centre.shape}')
    if velocity_edge.shape != velocity_centre.shape:
        raise ValueError(
            f'Centre/edge velocity shapes differ: {velocity_centre.shape} and '
            f'{velocity_edge.shape}')
    ux_centre = velocity_centre[0, 0]
    ux_edge = velocity_edge[0, 0]

    fig = plt.figure(figsize=(12, 6))
    grid = fig.add_gridspec(
        2, 4, width_ratios=(1, 1, 1, 0.055),
        left=0.10, right=0.92, bottom=0.10, top=0.82,
        wspace=0.28, hspace=0.25)
    axes = np.asarray([[fig.add_subplot(grid[row, col])
                        for col in range(3)] for row in range(2)])
    cax = fig.add_subplot(grid[:, 3])
    mid = ux_centre.shape[0] // 2
    titles = [f'$u_x$ (xy slice, z={mid})',
              f'$u_x$ (xz slice, y={mid})',
              f'$u_x$ (yz slice, x={mid})']
    slices_centre = [ux_centre[:, :, mid], ux_centre[:, mid, :],
                     ux_centre[mid, :, :]]
    slices_edge = [ux_edge[:, :, mid], ux_edge[:, mid, :],
                   ux_edge[mid, :, :]]
    vmin = min(item.min() for item in slices_centre + slices_edge)
    vmax = max(item.max() for item in slices_centre + slices_edge)
    margin = (vmax - vmin) * 0.03
    vmin -= margin
    vmax += margin

    for col in range(3):
        ax = axes[0, col]
        im = ax.imshow(slices_centre[col], cmap='viridis', aspect='auto',
                       vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(titles[col], fontsize=19)
        ax.tick_params(labelsize=17)
        if col == 0:
            ax.set_ylabel('Low-gradient (centre)', fontsize=17)

        ax = axes[1, col]
        im = ax.imshow(slices_edge[col], cmap='viridis', aspect='auto',
                       vmin=vmin, vmax=vmax, origin='lower')
        ax.tick_params(labelsize=17)
        if col == 0:
            ax.set_ylabel('High-gradient (edge)', fontsize=17)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Streamwise velocity $u_x$', fontsize=17)
    cbar.ax.tick_params(labelsize=15)
    fig.suptitle('Generated Homogeneous Isotropic Turbulence — Frame 0',
                 fontsize=20)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


# ─────────────────────────────────────────────────────────────
#  3.  Centre vs Edge accuracy bar chart
# ─────────────────────────────────────────────────────────────

def plot_region_comparison(region_accuracies, save_path=None):
    """Bar chart comparing accuracy across regions (centre, edge, etc.).

    Args:
        region_accuracies: dict like {'Centre': 85.0, 'Edge': 62.5}
    """
    labels = list(region_accuracies.keys())
    accs = list(region_accuracies.values())
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(labels)]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, accs, color=colors, alpha=0.75, width=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy by Region')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ─────────────────────────────────────────────────────────────
#  3.  Confusion matrix
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(model, dataloader, device='cpu', class_names=None, save_path=None):
    """Compute and plot classification confusion matrix."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            x = torch.cat([batch['velocity'], batch['pressure'].unsqueeze(1)], dim=1).to(device)
            labels = batch['label'].to(device)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    cm = confusion_matrix(all_labels, all_preds)
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', colorbar=True, values_format='d')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_confusion_matrix_from_data(all_labels, all_preds, class_names=None, save_path=None):
    """Plot confusion matrix from pre-computed predictions."""
    cm = confusion_matrix(all_labels, all_preds)
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', colorbar=True, values_format='d')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_per_class_with_confusion(all_labels, all_preds, n_classes=10, save_path=None):
    """Combined 1×2 figure: per-class accuracy (left) + confusion matrix (right).

    Replaces the standalone region-comparison bar chart.
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # Per-class accuracy
    correct = np.zeros(n_classes)
    total = np.zeros(n_classes)
    for c in range(n_classes):
        mask = all_labels == c
        total[c] = mask.sum()
        correct[c] = (all_preds[mask] == c).sum()
    accs = np.divide(correct, total, where=total > 0, out=np.full_like(correct, 0, dtype=float)) * 100

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.5, 5.2), constrained_layout=True)

    # Left: per-class accuracy (color by above/below mean of non-zero classes)
    nonzero_mean = accs[accs > 0].mean() if np.any(accs > 0) else 0
    bar_colors = ['#2c7bb6' if a >= nonzero_mean else '#d7191c' for a in accs]
    bars = ax1.bar(range(n_classes), accs, color=bar_colors, alpha=0.8, edgecolor='gray', linewidth=0.5)
    ax1.axhline(y=nonzero_mean, color='gray', ls='--', lw=1.5, label=f'Mean {nonzero_mean:.1f}%')
    ax1.set_xlabel('Velocity Class', fontsize=20)
    ax1.set_ylabel('Accuracy (%)', fontsize=20)
    ax1.set_title('(a) Per-Class Accuracy', fontsize=22)
    ax1.set_xticks(range(n_classes))
    ax1.tick_params(labelsize=20)
    ax1.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 1.0))
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 110)
    for bar, acc in zip(bars, accs):
        if acc > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{acc:.0f}%', ha='center', va='bottom', fontsize=17)

    # Right: confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(n_classes))
    class_names = [str(i) for i in range(n_classes)]
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax2, cmap='Blues', colorbar=True, values_format='d', text_kw={'fontsize': 17})
    ax2.set_title('(b) Confusion Matrix', fontsize=22)
    ax2.tick_params(labelsize=20)
    ax2.set_xlabel('Predicted Class', fontsize=20)
    ax2.set_ylabel('True Class', fontsize=20)
    # Resize colorbar ticks
    if disp.im_ and disp.im_.colorbar:
        disp.im_.colorbar.ax.tick_params(labelsize=17)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────
#  4.  Field reconstruction (decoded vs ground truth)
# ─────────────────────────────────────────────────────────────

def plot_field_reconstruction(model, decoder, dataloader, device='cpu',
                               n_samples=3, save_path=None):
    """Compare ground-truth fields with decoded reconstructions.

    Shows velocity magnitude slices for n_samples from the dataloader.
    True and predicted fields share the same color scale for fair comparison.
    """
    model.eval()
    decoder.eval()
    batch = next(iter(dataloader))
    x = torch.cat([batch['velocity'], batch['pressure'].unsqueeze(1)], dim=1).to(device)

    with torch.no_grad():
        logits, z_all = model(x, return_all_latents=True)
        z_T = z_all[-1]
        fields_pred = decoder(z_T).cpu().numpy()  # (B, 4, H, H, H)
        fields_true = x.cpu().numpy()[:n_samples]  # (B, 4, H, H, H)

    vel_mag_true = np.sqrt(np.sum(fields_true[:, :3] ** 2, axis=1))
    vel_mag_pred = np.sqrt(np.sum(fields_pred[:n_samples, :3] ** 2, axis=1))

    nrows = n_samples
    fig, axes = plt.subplots(nrows, 3, figsize=(12, 3 * nrows))
    # Ensure axes is always 2D for uniform indexing
    if nrows == 1:
        axes = axes.reshape(1, -1)

    titles = ['True |u|', 'Reconstructed |u|', 'Difference |Δu|']
    for row in range(nrows):
        mid_z = fields_true.shape[3] // 2
        u_true_slice = vel_mag_true[row, :, :, mid_z]
        u_pred_slice = vel_mag_pred[row, :, :, mid_z]
        diff_slice = np.abs(u_true_slice - u_pred_slice)

        # Column 0: ground truth
        ax = axes[row, 0]
        vmax_true = max(abs(u_true_slice.max()), abs(u_true_slice.min())) or 1
        im0 = ax.imshow(u_true_slice, cmap='viridis', aspect='auto',
                        vmin=0, vmax=vmax_true)
        ax.set_title(titles[0] if row == 0 else '')
        ax.set_ylabel('y')
        plt.colorbar(im0, ax=ax, fraction=0.046)

        # Column 1: predicted (separate color scale — may be small vs ground truth)
        ax = axes[row, 1]
        vmax_pred = u_pred_slice.max() or 1
        if u_pred_slice.max() - u_pred_slice.min() < 1e-12:
            # Degenerate: prediction is constant (near-zero from untrained decoder)
            im1 = ax.imshow(u_pred_slice, cmap='viridis', aspect='auto',
                            vmin=0, vmax=max(vmax_pred, 1e-6))
            ax.text(0.5, 0.5, 'Model not converged', ha='center', va='center',
                    transform=ax.transAxes, fontsize=11, color='gray')
        else:
            im1 = ax.imshow(u_pred_slice, cmap='viridis', aspect='auto',
                            vmin=0, vmax=vmax_pred)
        ax.set_title(titles[1] if row == 0 else '')
        plt.colorbar(im1, ax=ax, fraction=0.046)

        # Column 2: absolute difference
        ax = axes[row, 2]
        im2 = ax.imshow(diff_slice, cmap='Reds', aspect='auto')
        ax.set_title(titles[2] if row == 0 else '')
        ax.set_xlabel('x')
        plt.colorbar(im2, ax=ax, fraction=0.046)

    plt.suptitle('Field Reconstruction (z = 16 slice)', fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_field_reconstruction_from_data(fields_true, fields_pred, n_samples=3, save_path=None):
    """Plot field reconstruction from pre-computed numpy arrays.

    Args:
        fields_true: np.ndarray (B, 4, H, H, H) — ground truth fields [u_x, u_y, u_z, p]
        fields_pred: np.ndarray (B, 4, H, H, H) — decoded/reconstructed fields
        n_samples: number of samples to plot
    """
    n_samples = min(n_samples, len(fields_true), len(fields_pred))
    vel_mag_true = np.sqrt(np.sum(fields_true[:n_samples, :3] ** 2, axis=1))
    vel_mag_pred = np.sqrt(np.sum(fields_pred[:n_samples, :3] ** 2, axis=1))

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    titles = ['True |u|', 'Reconstructed |u|', 'Difference |Δu|']
    for row in range(n_samples):
        mid_z = fields_true.shape[3] // 2
        u_true_slice = vel_mag_true[row, :, :, mid_z]
        u_pred_slice = vel_mag_pred[row, :, :, mid_z]
        diff_slice = np.abs(u_true_slice - u_pred_slice)

        ax = axes[row, 0]
        vmax_true = max(abs(u_true_slice.max()), abs(u_true_slice.min())) or 1
        im0 = ax.imshow(u_true_slice, cmap='viridis', aspect='auto',
                        vmin=0, vmax=vmax_true)
        ax.set_title(titles[0] if row == 0 else '', fontsize=19)
        ax.set_ylabel('y', fontsize=17)
        ax.tick_params(labelsize=17)
        cbar = plt.colorbar(im0, ax=ax, fraction=0.046)
        cbar.ax.tick_params(labelsize=17)

        ax = axes[row, 1]
        vmax_pred = max(abs(u_pred_slice.max()), abs(u_pred_slice.min())) or 1
        im1 = ax.imshow(u_pred_slice, cmap='viridis', aspect='auto',
                        vmin=0, vmax=vmax_pred)
        ax.set_title(titles[1] if row == 0 else '', fontsize=19)
        ax.tick_params(labelsize=17)
        cbar = plt.colorbar(im1, ax=ax, fraction=0.046)
        cbar.ax.tick_params(labelsize=17)

        ax = axes[row, 2]
        im2 = ax.imshow(diff_slice, cmap='Reds', aspect='auto')
        ax.set_title(titles[2] if row == 0 else '', fontsize=19)
        ax.set_xlabel('x', fontsize=17)
        ax.set_ylabel('y', fontsize=17)
        ax.tick_params(labelsize=17)
        cbar = plt.colorbar(im2, ax=ax, fraction=0.046)
        cbar.ax.tick_params(labelsize=17)

    plt.suptitle('Field Reconstruction (z = 16 slice)', fontsize=20)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ─────────────────────────────────────────────────────────────
#  5.  SPIDER-discovered equation coefficient bar chart
# ─────────────────────────────────────────────────────────────

def _sanitize_latex(label):
    """Convert Unicode math symbols to LaTeX for matplotlib's math text engine."""
    replacements = [
        ('∇²', r'\nabla^2'),
        ('∇', r'\nabla '),
        ('·', r'\cdot '),
        ('∂', r'\partial '),
        ('₁', r'_1'), ('₂', r'_2'), ('₃', r'_3'), ('₄', r'_4'), ('₅', r'_5'),
        ('²', r'^2'),
    ]
    result = label
    for old, new in replacements:
        result = result.replace(old, new)
    math_chars = {'∇', '·', '∂', '₁', '₂', '₃', '₄', '²'}
    if any(c in label for c in math_chars):
        return f'${result}$'
    return result


def plot_spider_coefficients(coefficients, term_descriptions, title=None, save_path=None, xlim=None):
    """Horizontal bar chart of SPIDER-discovered equation coefficients.

    Args:
        coefficients: 1D array of coefficient values
        term_descriptions: list of term name strings
    """
    coeffs = np.asarray(coefficients).flatten()
    terms = term_descriptions

    # Sort by absolute magnitude
    idx = np.argsort(np.abs(coeffs))[::-1]
    coeffs = coeffs[idx]
    terms = [terms[i] for i in idx]

    # Convert Unicode math symbols to LaTeX for proper rendering
    terms_latex = [_sanitize_latex(t) for t in terms]

    colors = ['#e74c3c' if c >= 0 else '#3498db' for c in coeffs]
    fig, ax = plt.subplots(figsize=(8, max(4, len(terms) * 0.35)))
    bars = ax.barh(range(len(terms)), coeffs, color=colors, alpha=0.75)
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(terms_latex, fontsize=21)
    ax.tick_params(axis='x', labelsize=22)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel('Coefficient', fontsize=22)
    ax.set_title(title or 'SPIDER Equation Coefficients', fontsize=24)
    ax.grid(True, alpha=0.3, axis='x')

    for bar, c in zip(bars, coeffs):
        ax.text(bar.get_width() + (0.02 if c >= 0 else -0.02), bar.get_y() + bar.get_height() / 2,
                f'{c:.4f}', va='center', ha='left' if c >= 0 else 'right', fontsize=20)

    if xlim is not None:
        ax.set_xlim(xlim)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ─────────────────────────────────────────────────────────────
#  6.  Per-class accuracy bar chart
# ─────────────────────────────────────────────────────────────

def plot_per_class_accuracy(model, dataloader, n_classes=10, device='cpu', save_path=None):
    """Bar chart of accuracy per class."""
    model.eval()
    correct = np.zeros(n_classes)
    total = np.zeros(n_classes)

    with torch.no_grad():
        for batch in dataloader:
            x = torch.cat([batch['velocity'], batch['pressure'].unsqueeze(1)], dim=1).to(device)
            labels = batch['label'].to(device)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            for c in range(n_classes):
                mask = labels == c
                total[c] += mask.sum().item()
                correct[c] += (preds[mask] == c).sum().item()

    accs = np.divide(correct, total, where=total > 0, out=np.full_like(correct, 0, dtype=float)) * 100

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ['#3498db' if a >= accs.mean() else '#e74c3c' for a in accs]
    bars = ax.bar(range(n_classes), accs, color=colors, alpha=0.7)
    ax.axhline(y=accs.mean(), color='gray', ls='--', lw=1, label=f'Mean {accs.mean():.1f}%')
    ax.set_xlabel('Class')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Class Accuracy')
    ax.set_xticks(range(n_classes))
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{acc:.0f}%', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_per_class_accuracy_from_data(all_labels, all_preds, n_classes=10, save_path=None):
    """Plot per-class accuracy bar chart from pre-computed predictions."""
    correct = np.zeros(n_classes)
    total = np.zeros(n_classes)
    for c in range(n_classes):
        mask = all_labels == c
        total[c] = mask.sum()
        correct[c] = (all_preds[mask] == c).sum()
    accs = np.divide(correct, total, where=total > 0, out=np.full_like(correct, 0, dtype=float)) * 100

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ['#3498db' if a >= accs.mean() else '#e74c3c' for a in accs]
    bars = ax.bar(range(n_classes), accs, color=colors, alpha=0.7)
    ax.axhline(y=accs.mean(), color='gray', ls='--', lw=1, label=f'Mean {accs.mean():.1f}%')
    ax.set_xlabel('Class')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Class Accuracy')
    ax.set_xticks(range(n_classes))
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{acc:.0f}%', ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ─────────────────────────────────────────────────────────────
#  7.  η_NS evolution during training
# ─────────────────────────────────────────────────────────────

def plot_eta_ns(eta_history, save_path=None):
    """Plot the relative NS residual η_NS over training epochs.

    η_NS = ||R_NS||_weak / (||∂_t u|| + ||(u·∇)u|| + ||∇p|| + ν||∇²u||)

    Lower η_NS means the reconstructed fields better satisfy the NS equation.
    """
    epochs = np.arange(1, len(eta_history) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, eta_history, 'm-', lw=2, marker='o', ms=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('η_NS')
    ax.set_title('Relative NS Residual η_NS')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    if len(eta_history) > 2:
        z = np.polyfit(epochs, np.log(eta_history), 1)
        ax.plot(epochs, np.exp(z[0] * epochs + z[1]), '--', color='gray', alpha=0.5,
                label=f'Decay rate {z[0]:.3f}/epoch')
        ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ─────────────────────────────────────────────────────────────
#  8.  Noise robustness (multi-method comparison)
# ─────────────────────────────────────────────────────────────

def plot_noise_robustness(noise_levels, results_dict, save_path=None):
    """Fig 1: Two-panel noise robustness — accuracy (left) and η_NS (right).

    Args:
        noise_levels: list of sigma values
        results_dict: {'MethodName': {'accuracy': [...], 'eta_ns': [...]}}
    """
    colors = {'CNN (BP)': '#95a5a6', 'NoProp (vanilla)': '#3498db',
              'PINN-style': '#e67e22', 'PI-NoProp (ours)': '#e74c3c'}
    markers = {'CNN (BP)': 's', 'NoProp (vanilla)': 'o',
               'PINN-style': '^', 'PI-NoProp (ours)': 'D'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x_labels = [f'{int(s*100)}%' if s < 1 else f'{int(s*100)}%' for s in noise_levels]
    # Show well-spaced tick labels to avoid overlap
    tick_indices = [0, 4, 5, 6]
    show_labels = [x_labels[i] if i in tick_indices else '' for i in range(len(x_labels))]

    for method, data in results_dict.items():
        color = colors.get(method, '#333333')
        marker = markers.get(method, 'o')
        ax1.plot(noise_levels, data['accuracy'], f'-{marker}',
                color=color, lw=2, ms=6, label=method)
        if 'eta_ns' in data:
            ax2.plot(noise_levels, data['eta_ns'], f'-{marker}',
                    color=color, lw=2, ms=6, label=method)

    ax1.set_xlabel('Noise Level $\\sigma$', fontsize=23)
    ax1.set_ylabel('Accuracy (%)', fontsize=23)
    ax1.set_title('(a) Classification Accuracy', fontsize=25)
    ax1.set_xticks(noise_levels)
    ax1.set_xticklabels(show_labels, rotation=45, fontsize=23)
    ax1.tick_params(axis='y', labelsize=23)
    ax1.set_ylim(0, 100)
    ax1.legend(fontsize=21)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Noise Level $\\sigma$', fontsize=23)
    ax2.set_ylabel('$\\eta_{\\mathrm{NS}}$', fontsize=23)
    ax2.set_title('(b) Physical Consistency', fontsize=25)
    ax2.set_xticks(noise_levels)
    ax2.set_xticklabels(show_labels, rotation=45, fontsize=23)
    ax2.tick_params(axis='y', labelsize=23)
    ax2.set_yscale('log')
    ax2.legend(fontsize=21)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────
#  9.  Lambda trade-off scatter plot
# ─────────────────────────────────────────────────────────────

def plot_lambda_tradeoff(lambda_values, accuracies, eta_ns_values, save_path=None):
    """Fig 2: Two-panel hyperparameter sensitivity — accuracy vs λ (left), η_NS vs λ (right).

    The left panel shows classification performance as a function of the physics
    loss weight; the right panel shows the resulting physics consistency.
    """
    lams = np.asarray(lambda_values)
    accs = np.asarray(accuracies)
    etas = np.asarray(eta_ns_values)

    # Sort by λ for connected lines
    idx = np.argsort(lams)
    lams_sorted = lams[idx]
    accs_sorted = accs[idx]
    etas_sorted = etas[idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Shared colour mapping
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(lams)))

    # ── Panel (a): Accuracy vs λ ──
    for i in idx:
        ax1.semilogx(lams[i], accs[i], 'o', color=colors[i], ms=10,
                     markeredgecolor='black', markeredgewidth=0.5, zorder=5)
    ax1.semilogx(lams_sorted, accs_sorted, '--', color='gray', alpha=0.4, lw=1)
    # Mean accuracy reference line
    mean_acc = accs.mean()
    ax1.axhline(y=mean_acc, color='gray', ls=':', lw=1.5, alpha=0.7,
                label=f'Mean {mean_acc:.1f}%')
    # Best λ annotation
    best_i = np.argmax(accs)
    ax1.annotate(f'$\\lambda$={lams[best_i]:.0e}\n(best legacy run)',
                 (lams[best_i], accs[best_i]),
                 xytext=(10, -30), textcoords='offset points',
                 fontsize=12, fontweight='bold', color='darkgreen',
                 arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    ax1.set_xlabel('Physics Loss Weight $\\lambda$', fontsize=16)
    ax1.set_ylabel('Validation Accuracy (%)', fontsize=16)
    ax1.set_title('(a) Accuracy vs $\\lambda$', fontsize=18)
    ax1.tick_params(labelsize=13)
    acc_span = max(float(np.ptp(accs_sorted)), 1.0)
    acc_pad = max(2.0, 0.18 * acc_span)
    ax1.set_ylim(max(0.0, float(accs_sorted.min()) - acc_pad),
                 min(100.0, float(accs_sorted.max()) + acc_pad))
    ax1.legend(fontsize=12, loc='best')
    ax1.grid(True, alpha=0.3)

    # ── Panel (b): η_NS vs λ ──
    for i in idx:
        ax2.semilogx(lams[i], etas[i], 's', color=colors[i], ms=10,
                   markeredgecolor='black', markeredgewidth=0.5, zorder=5)
    ax2.semilogx(lams_sorted, etas_sorted, '--', color='gray', alpha=0.4, lw=1)
    ax2.set_xlabel('Physics Loss Weight $\\lambda$', fontsize=16)
    ax2.set_ylabel('$\\eta_{\\mathrm{NS}}$ (lower is better)', fontsize=16)
    ax2.set_title('(b) Physical Consistency vs $\\lambda$', fontsize=18)
    ax2.tick_params(labelsize=13)
    eta_span = max(float(np.ptp(etas_sorted)), 1e-3)
    eta_pad = max(0.005, 0.18 * eta_span)
    ax2.set_ylim(float(etas_sorted.min()) - eta_pad,
                 float(etas_sorted.max()) + eta_pad)
    ax2.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  10.  t-SNE latent space visualisation
# ─────────────────────────────────────────────────────────────

def plot_tsne(latents, labels, title='t-SNE of Latent Space', save_path=None,
              secondary_color=None, secondary_label='Mean $|\\mathbf{u}|$'):
    """Fig 7: t-SNE projection of NoProp latent codes.

    Creates a 1×2 figure when secondary_color is provided:
      (a) coloured by discrete class labels
      (b) coloured by a continuous physical quantity (e.g. velocity magnitude)
    Falls back to single-panel when secondary_color is None.

    Args:
        latents: (N, D) numpy array of latent vectors
        labels: (N,) integer class labels
        secondary_color: (N,) continuous values for second panel
        secondary_label: colorbar label for the second panel
    """
    from sklearn.manifold import TSNE

    n_panels = 2 if secondary_color is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5.5))
    if n_panels == 1:
        axes = [axes]

    # Compute t-SNE once
    perplexity = min(30, len(latents) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, verbose=0)
    z_2d = tsne.fit_transform(latents)

    # Panel (a): coloured by class
    ax = axes[0]
    scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap='tab10',
                         s=60, alpha=0.8, edgecolors='none')
    ax.set_title('(a) Coloured by Velocity Class', fontsize=27)
    ax.set_xlabel('t-SNE dim 1', fontsize=25)
    ax.set_ylabel('t-SNE dim 2', fontsize=25)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046)
    cbar.set_label('Velocity Class', fontsize=25)
    cbar.ax.tick_params(labelsize=25)

    # Panel (b): coloured by physical quantity
    if secondary_color is not None:
        ax = axes[1]
        scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=secondary_color,
                             cmap='viridis', s=60, alpha=0.8, edgecolors='none')
        ax.set_title(f'(b) Coloured by {secondary_label}', fontsize=27)
        ax.set_xlabel('t-SNE dim 1', fontsize=25)
        ax.set_ylabel('t-SNE dim 2', fontsize=25)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046)
        cbar.set_label(secondary_label, fontsize=25)
        cbar.ax.tick_params(labelsize=25)

    plt.suptitle(title, fontsize=28)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────
#  11.  All-in-one summary figure
# ─────────────────────────────────────────────────────────────

def plot_summary_dashboard(history, region_accuracies=None,
                            spider_eqs=None, noise_data=None, save_path=None):
    """Produce a large summary figure with multiple panels."""
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    epochs = np.arange(1, len(history['train_loss']) + 1)

    # (1,1) Training loss
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(epochs, history['train_loss'], 'b-', lw=2, label='Train')
    if 'val_loss' in history:
        ax.plot(epochs, history['val_loss'], 'r--', lw=2, label='Val')
    ax.set_title('Loss')
    ax.set_yscale('log')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (1,2) Accuracy
    ax = fig.add_subplot(gs[0, 1])
    if 'val_accuracy' in history:
        ax.plot(epochs, history['val_accuracy'], 'g-', lw=2)
    ax.set_title('Accuracy (%)')
    ax.grid(True, alpha=0.3)

    # (1,3) Region comparison
    ax = fig.add_subplot(gs[0, 2])
    if region_accuracies:
        labels = list(region_accuracies.keys())
        accs = list(region_accuracies.values())
        colors = ['#3498db', '#e74c3c', '#2ecc71'][:len(labels)]
        ax.bar(labels, accs, color=colors, alpha=0.7)
        ax.set_title('Region Accuracy')
        ax.grid(True, alpha=0.3, axis='y')

    # (2,1) Loss components
    ax = fig.add_subplot(gs[1, 0])
    has_any = False
    for key, color in [('cls_loss', 'orange'), ('diff_loss', 'purple'), ('phys_loss', 'brown')]:
        if key in history and len(history[key]) > 0:
            ax.plot(epochs, history[key], 'o-', color=color, lw=1.5, ms=3, label=key)
            has_any = True
    ax.set_title('Loss Components')
    if has_any:
        ax.set_yscale('log')
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (2,2) Physics loss
    ax = fig.add_subplot(gs[1, 1])
    if 'phys_ns' in history:
        ax.plot(epochs, history['phys_ns'], 'm-', lw=1.5, label='NS')
    if 'phys_cont' in history:
        ax.plot(epochs, history['phys_cont'], 'c-', lw=1.5, label='Cont')
    if 'phys_ns' in history or 'phys_cont' in history:
        ax.set_title('Physics Loss')
        ax.set_yscale('log')
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (2,3) η_NS
    ax = fig.add_subplot(gs[1, 2])
    if 'eta_ns' in history:
        ax.plot(epochs, history['eta_ns'], 'm-', lw=2)
        ax.set_title('η_NS')
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # (3,1) SPIDER coefficients
    ax = fig.add_subplot(gs[2, 0])
    if spider_eqs is not None:
        coeffs = np.asarray(spider_eqs[0]).flatten()
        terms = spider_eqs[1]
        idx = np.argsort(np.abs(coeffs))[::-1][:min(8, len(coeffs))]
        colors = ['#e74c3c' if coeffs[i] >= 0 else '#3498db' for i in idx]
        ax.barh(range(len(idx)), coeffs[idx], color=colors, alpha=0.7)
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([_sanitize_latex(terms[i]) for i in idx], fontsize=7)
        ax.axvline(0, color='black', lw=0.5)
        ax.set_title('SPIDER Coefficients')
        ax.grid(True, alpha=0.3, axis='x')
    else:
        ax.text(0.5, 0.5, 'No SPIDER data', ha='center', va='center', transform=ax.transAxes)

    # (3,2) Noise robustness
    ax = fig.add_subplot(gs[2, 1])
    if noise_data is not None:
        ax.plot(noise_data[0], noise_data[1], 'o-', color='#2ecc71', lw=2)
        ax.set_title('Noise Robustness')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No noise data', ha='center', va='center', transform=ax.transAxes)

    # (3,3) Per-class accuracy
    ax = fig.add_subplot(gs[2, 2])
    if 'per_class_acc' in history:
        accs = history['per_class_acc']
        ax.bar(range(len(accs)), accs, alpha=0.7)
        ax.set_title('Per-Class Accuracy')
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('PI-NoProp Training Dashboard', fontsize=14, fontweight='bold')
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ─────────────────────────────────────────────────────────────
#  Demo: generate sample data and show all plots
# ─────────────────────────────────────────────────────────────

def demo():
    """Run all plotting functions with fake data to verify they work."""
    set_global_style()
    np.random.seed(42)

    # Fake training history
    n_epochs = 20
    history = {
        'train_loss': np.exp(-0.2 * np.arange(n_epochs)) + 0.1 * np.random.rand(n_epochs),
        'val_loss': np.exp(-0.15 * np.arange(n_epochs)) + 0.15 * np.random.rand(n_epochs),
        'val_accuracy': 100 * (1 - np.exp(-0.1 * np.arange(n_epochs))) + 2 * np.random.randn(n_epochs),
        'cls_loss': 0.5 * np.exp(-0.2 * np.arange(n_epochs)) + 0.05 * np.random.rand(n_epochs),
        'diff_loss': 1.0 * np.exp(-0.15 * np.arange(n_epochs)) + 0.1 * np.random.rand(n_epochs),
        'phys_loss': 1e-4 * np.exp(-0.3 * np.arange(n_epochs)) + 1e-6 * np.random.rand(n_epochs),
        'phys_ns': 1e-4 * np.exp(-0.3 * np.arange(n_epochs)),
        'phys_cont': 5e-5 * np.exp(-0.25 * np.arange(n_epochs)),
        'eta_ns': 0.5 * np.exp(-0.2 * np.arange(n_epochs)),
        'per_class_acc': 100 * np.random.beta(5, 2, 10),
    }
    # Clip
    history['val_accuracy'] = np.clip(history['val_accuracy'], 0, 100)

    print('=== 1. Training Curves ===')
    plot_training_curves(history)

    print('=== 2. Region Comparison ===')
    plot_region_comparison({'Centre': 85.3, 'Edge': 62.7})

    print('=== 3. SPIDER Coefficients ===')
    coeffs = np.array([1.0, 1.0, 1.0, -5e-5])
    terms = ['∂_t u', '(u·∇)u', '∇p', '∇²u']
    plot_spider_coefficients(coeffs, terms)

    print('=== 4. Per-Class Accuracy ===')
    fig, ax = plt.subplots(figsize=(8, 4))
    accs = 100 * np.random.beta(5, 2, 10)
    ax.bar(range(10), accs, color=['#3498db' if a >= accs.mean() else '#e74c3c' for a in accs], alpha=0.7)
    ax.axhline(y=accs.mean(), color='gray', ls='--', lw=1, label=f'Avg {accs.mean():.1f}%')
    ax.set_xlabel('Class')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Class Classification Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

    print('=== 5. η_NS Evolution ===')
    plot_eta_ns(history['eta_ns'])

    print('=== 6. Noise Robustness ===')
    plot_noise_robustness(
        [0.0, 0.01, 0.05, 0.1, 0.2],
        {'PI-NoProp (ours)': {'accuracy': [85.3, 82.1, 76.4, 65.8, 48.2],
                              'eta_ns': [0.01, 0.015, 0.02, 0.03, 0.05]}},
    )

    print('=== 7. Summary Dashboard ===')
    plot_summary_dashboard(history, {'Centre': 85.3, 'Edge': 62.7},
                            spider_eqs=(coeffs, terms),
                            noise_data=([0.0, 0.01, 0.05, 0.1], [85.3, 82.1, 76.4, 65.8]))

    print('=== Demo complete ===')


if __name__ == '__main__':
    demo()

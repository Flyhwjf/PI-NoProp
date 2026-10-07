"""Render the main-text and appendix figures via matched standalone scripts.

The CLI defaults to the v5 input32/target16 protocol. Historical drawing
functions remain available for compatibility, but main never calls them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT/'paper/figures'
FIGURE_CODE = ROOT/'paper/figure_code'
if str(FIGURE_CODE) not in sys.path:
    sys.path.insert(0, str(FIGURE_CODE))

from figure_protocol import (FigureProtocol, PROTOCOLS, V5, load_aggregate,
                             read_json, validate_metadata)

# Manuscript order. False also marks the current-code architecture schematic,
# whose geometry is fixed to v5; its script emits a clearly labelled diagram.
PAPER_FIGURES = (
    ('fig_framework', False),
    ('fig_data_samples', True),
    ('fig_ablation', True),
    ('fig_noise', True),
    ('fig_latent_metrics', True),
    ('fig_dns_quality', False),
    ('fig_training_convergence', True),
    ('fig_spider_noise', False),
    ('fig_latent_analysis', True),
)
PAPER_AGGREGATES = ('results', 'noise', 'lambda_ablation',
                    'decoder_ablation', 'latent_analysis')
REGIONS = ('low_enstrophy', 'high_enstrophy')
SEEDS = (42, 123, 456)

COLORS = {'none': '#8b95a5', 'analytic': '#e99b42', 'discovered': '#2b7bbb',
          'green': '#3a9d72', 'red': '#c44e52', 'navy': '#274c77'}


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': 0.18, 'figure.dpi': 150,
    })


def box_axes(axis):
    """Use a visible, consistent four-sided frame for a manuscript panel."""
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color('#334155')
        spine.set_linewidth(.85)


def place_panel_label_left_of_title(fig, axis, label):
    """Place a panel marker immediately to the left of the centered title."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title_bbox = axis.title.get_window_extent(renderer=renderer)
    marker = axis.text(0, 0, label, transform=axis.transAxes,
                       ha='right', va='center', fontsize=10.5,
                       fontweight='bold', clip_on=False)
    marker.set_in_layout(False)
    fig.canvas.draw()
    marker_bbox = marker.get_window_extent(renderer=renderer)
    x_px = title_bbox.x0 - marker_bbox.width - 5
    y_px = title_bbox.y0 + .5 * title_bbox.height
    x_axes, y_axes = axis.transAxes.inverted().transform((x_px, y_px))
    marker.set_position((x_axes, y_axes))


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES/name, dpi=240, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def framework():
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    boxes = [
        (0.02, '15 independent\ndecaying-HIT trajectories', '#dbeafe'),
        (0.22, '4-D weak SPIDER\n5 candidates', '#e0f2fe'),
        (0.42, 'Validated full NS\nartifact', '#dcfce7'),
        (0.62, 'NS energy-rate condition\n+ temporal decoder', '#fef3c7'),
        (0.82, 'Consistent local\nNoProp blocks', '#fce7f3'),
    ]
    for x, label, color in boxes:
        patch = FancyBboxPatch((x, .35), .16, .30,
                               boxstyle='round,pad=.02,rounding_size=.025',
                               facecolor=color, edgecolor='white', linewidth=1.5)
        ax.add_patch(patch); ax.text(x+.08, .50, label, ha='center', va='center')
    for x in (.18, .38, .58, .78):
        ax.add_patch(FancyArrowPatch((x, .50), (x+.04, .50), arrowstyle='-|>',
                                    mutation_scale=13, color=COLORS['navy']))
    ax.text(.30, .18, '9 discovery / 3 validation / 3 test trajectories',
            ha='center', color='#475569')
    ax.text(.70, .18, r'$\partial_tu+c_2(u\cdot\nabla)u+c_3\nabla p+c_4\nabla^2u$',
            ha='center', color='#475569')
    save(fig, 'fig_framework.pdf')


def local_update():
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    items = [
        (.02, .58, .16, .22, '3-D condition encoder\n+ NS energy rate', '#39739d'),
        (.02, .18, .16, .22, 'Noisy target\nlatent', '#4695b5'),
        (.28, .38, .18, .27, 'Sampled block $J$\nonly optimizer stepped', '#3d9967'),
        (.55, .55, .18, .25, 'Frozen temporal decoder\n$9\\times4\\times16^3$', '#dd762d'),
        (.55, .16, .18, .24, 'Local objective\n$T(\\mathcal{L}_{diff}+\\lambda\\mathcal{L}_{NS})$', '#c74e53'),
        (.81, .55, .17, .25, 'Prototype readout\nfrom $z_T$', '#39739d'),
    ]
    for x,y,w,h,label,color in items:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.018',
                                   facecolor=color,edgecolor='white'))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',color='white',weight='bold')
    arrows=[((.18,.69),(.28,.55),'#274c77'),((.18,.29),(.28,.47),'#274c77'),
            ((.46,.52),(.55,.67),'#3d9967'),((.64,.55),(.64,.40),'#dd762d'),
            ((.55,.28),(.46,.45),'#c74e53'),((.73,.67),(.81,.67),'#274c77')]
    for start,end,color in arrows:
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=13,
                                    linewidth=1.7,color=color))
    ax.text(.37,.18,'All other blocks: no graph, no gradient',ha='center',color='#64748b')
    ax.text(.895,.28,'No readout parameters\nor cross-block gradient',ha='center',
            color=COLORS['red'],weight='bold')
    ax.set_title('Strictly local full-NS block update',fontsize=15,weight='bold')
    save(fig, 'fig_local_training.pdf')


def dns_quality(manifest):
    records = sorted(manifest['trajectories'], key=lambda item: item['trajectory_id'])
    diagnostics = [record['diagnostics'] for record in records]
    time = np.asarray([item['time'] for item in diagnostics[0]])
    time -= time[0]

    def normalised_series(name):
        values = np.asarray([[item[name] for item in trajectory]
                             for trajectory in diagnostics], dtype=float)
        return values/values[:, :1]

    energy = normalised_series('kinetic_energy')
    dissipation = normalised_series('dissipation')
    reynolds = normalised_series('re_lambda')
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.0),
                             gridspec_kw={'width_ratios': (1.16, 1)},
                             constrained_layout=True)
    for axis in axes:
        box_axes(axis)

    ax = axes[0]
    for curve in energy:
        ax.plot(time, curve, color='#8ab6d6', alpha=.28, linewidth=.75)
    ax.fill_between(time, energy.min(axis=0), energy.max(axis=0),
                    color='#6baed6', alpha=.18, linewidth=0,
                    label='Energy range across trajectories')
    ax.plot(time, energy.mean(axis=0), color=COLORS['navy'], linewidth=2.4,
            label=r'Mean kinetic energy $K/K_0$')
    ax.plot(time, dissipation.mean(axis=0), color=COLORS['red'], linewidth=2.1,
            label=r'Mean dissipation $\varepsilon/\varepsilon_0$')
    ax.plot(time, reynolds.mean(axis=0), color=COLORS['green'], linewidth=2.1,
            label=r'Mean $Re_\lambda/Re_{\lambda,0}$')
    ax.axhline(1, color='#94a3b8', linewidth=.8, linestyle=':')
    lower = min(energy.min(), reynolds.min())-.0015
    upper = dissipation.max()+.0015
    ax.set(xlabel=r'Stored-window time $t-t_0$',
           ylabel='Normalised trajectory statistic',
           ylim=(lower, upper), title='Resolved decay dynamics')
    mean_drop = 100*(1-energy[:, -1]).mean()
    mean_re_drop = 100*(1-reynolds[:, -1]).mean()
    mean_diss_growth = 100*(dissipation[:, -1]-1).mean()
    ax.text(.03, .93,
            (f'$K$ drop: {mean_drop:.2f}%\n'
             f'$Re_\\lambda$ drop: {mean_re_drop:.2f}%\n'
             f'$\\varepsilon$ increase: {mean_diss_growth:.2f}%'),
            transform=ax.transAxes, fontsize=8.2, va='top',
            bbox={'boxstyle': 'round,pad=.35', 'facecolor': 'white',
                  'edgecolor': '#cbd5e1', 'alpha': .92})
    ax.legend(frameon=True, facecolor='white', edgecolor='#94a3b8',
              framealpha=.92, fancybox=False, loc='upper right', fontsize=7.8)

    ax = axes[1]
    trajectory_ids = np.asarray([record['trajectory_id'] for record in records])
    divergence = np.asarray([record['quality']['max_divergence_rms']
                             for record in records])
    cfl = np.asarray([record['quality']['max_cfl'] for record in records])
    split_bands = [(-.5, 8.5, '#dbeafe', 'Discovery (9)'),
                   (8.5, 11.5, '#fef3c7', 'Validation (3)'),
                   (11.5, 14.5, '#dcfce7', 'Test (3)')]
    for left, right, colour, label in split_bands:
        ax.axvspan(left, right, color=colour, alpha=.58, zorder=0)
        ax.text((left+right)/2, .965, label, ha='center', va='top',
                transform=ax.get_xaxis_transform(), fontsize=7.6,
                color='#475569', fontweight='bold')
    divergence_line, = ax.plot(
        trajectory_ids, divergence, color=COLORS['navy'], marker='o',
        markersize=4.5, linewidth=1.5, label='Maximum divergence RMS')
    ax.set(xlabel='Trajectory ID', ylabel='Maximum divergence RMS',
           xlim=(-.5, 14.5), xticks=np.arange(0, 15, 2),
           ylim=(0, divergence.max()*1.24))
    ax.set_title('Numerical quality and split isolation', pad=17)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax2 = ax.twinx()
    cfl_line, = ax2.plot(
        trajectory_ids, cfl, color='#dd762d', marker='s', markersize=4.2,
        linewidth=1.4, label='Maximum CFL')
    limit_line = ax2.axhline(.5, color=COLORS['red'], linestyle='--',
                             linewidth=1.4, label='CFL quality limit')
    ax2.set_ylabel('Maximum CFL', color='#9a4e16')
    ax2.tick_params(axis='y', colors='#9a4e16')
    ax2.set_ylim(0, .56)
    ax2.spines['right'].set_visible(True)
    ax.legend([divergence_line, cfl_line, limit_line],
              [item.get_label() for item in
               (divergence_line, cfl_line, limit_line)],
              frameon=True, facecolor='white', edgecolor='#94a3b8',
              framealpha=.92, fancybox=False, loc='center left', fontsize=7.8)
    ax.text(.97, .40,
            (f'max div. = {divergence.max():.2e}\n'
             f'max CFL = {cfl.max():.3f} ({100*cfl.max()/.5:.1f}% of limit)'),
            transform=ax.transAxes, ha='right', va='center', fontsize=8.2,
            bbox={'boxstyle': 'round,pad=.35', 'facecolor': 'white',
                  'edgecolor': '#cbd5e1', 'alpha': .92})
    place_panel_label_left_of_title(fig, axes[0], '(a)')
    place_panel_label_left_of_title(fig, axes[1], '(b)')
    save(fig, 'fig_dns_quality.pdf')


def generate_data_samples():
    import runpy

    runpy.run_path(
        str(ROOT / 'paper/figure_code/fig_data_samples.py'),
        run_name='__main__',
    )


def spider_figure(artifact):
    with plt.rc_context({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
            'font.size': 10.5,
            'axes.titlesize': 11.5,
            'axes.labelsize': 10.5,
            'xtick.labelsize': 9.5,
            'ytick.labelsize': 9.5}):
        _spider_figure(artifact)


def _spider_figure(artifact):
    expected = np.asarray(
        artifact['expected_equation_for_post_discovery_audit']['coefficients'],
        dtype=float)
    discovered = np.asarray(artifact['equation']['coefficients'], dtype=float)
    metrics = artifact['metrics']
    bootstrap_std = np.asarray(metrics['bootstrap_coefficient_std'], dtype=float)
    labels = [r'$\partial_t\mathbf{u}$', r'$(\mathbf{u}\!\cdot\!\nabla)\mathbf{u}$',
              r'$\nabla p$', r'$\nabla^2\mathbf{u}$']
    fig, ax = plt.subplots(figsize=(8.6, 4.9), constrained_layout=False)
    fig.subplots_adjust(left=.13, right=.97, bottom=.15, top=.90)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#334155')
        spine.set_linewidth(.85)

    # The main plot uses the residual scale; coefficient recovery is shown in
    # a compact inset so the two metrics retain their distinct units.
    residuals = np.asarray([
        metrics['discovery_eta'], metrics['validation_eta'], metrics['test_eta'],
        metrics['next_best_validation_eta']])
    x = np.arange(4)
    threshold = artifact['config']['max_validation_eta']
    ax.axhspan(5e-4, threshold, color='#dcfce7', alpha=.16, zorder=0)
    ax.axhline(threshold, color='#d97706', linestyle='--', linewidth=1.35,
               label=fr'Acceptance threshold ($\eta={threshold:g}$)')
    ax.plot(x[:3], residuals[:3], color=COLORS['green'], linewidth=2.0,
            marker='o', markersize=7, label='Accepted NS support', zorder=3)
    ax.scatter(x[3], residuals[3], marker='D', s=62, color=COLORS['red'],
               label='Nearest alternative', zorder=3)
    for index, value in enumerate(residuals):
        ax.annotate(f'{value:.2e}', (index, value), xytext=(0, 8),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=8.5, color='#334155')
    ax.text(2.22, 1.65e-2,
            fr'{metrics["support_separation_ratio"]:.1f}$\times$ above test',
            color=COLORS['red'], fontsize=9.0, ha='center', va='center')
    ax.set_yscale('log')
    ax.set_ylim(6e-4, 7e-2)
    ax.set_xlim(-.38, 3.38)
    ax.set_xticks(x, ['Discovery', 'Validation', 'Test',
                      'Nearest\nalternative'])
    ax.set(xlabel='Trajectory split / candidate support',
           ylabel=r'Contribution-normalised weak residual $\eta$',
           title='SPIDER equation recovery and trajectory-disjoint validation')
    ax.grid(axis='y', which='both', alpha=.18)
    ax.grid(axis='x', visible=False)
    ax.legend(frameon=False, loc='upper right', fontsize=8.5, ncol=2,
              handlelength=1.8, columnspacing=1.0)

    # Compact coefficient-recovery inset.
    relative_deviation = 100*(discovered/expected-1)
    relative_std = 100*np.abs(bootstrap_std/expected)
    y = np.arange(len(labels))
    inset = ax.inset_axes([.13, .62, .36, .25], facecolor='white')
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_color('#64748b')
        spine.set_linewidth(.7)
    inset.axvline(0, color='#64748b', linewidth=1.0, linestyle='--')
    inset.errorbar(relative_deviation[1:], y[1:], xerr=relative_std[1:],
                   fmt='o', markersize=5.5, capsize=2.5, elinewidth=1.0,
                   color=COLORS['discovered'], ecolor='#4f94c5', zorder=3)
    inset.scatter(relative_deviation[0], y[0], marker='D', s=38,
                  color=COLORS['none'], zorder=3)
    for index, (deviation, coefficient) in enumerate(
            zip(relative_deviation, discovered)):
        annotation = ('fixed at 1' if index == 0
                      else f'{coefficient:.4f}')
        inset.text(deviation+.012, index-.11, annotation, ha='left',
                   va='center', fontsize=7.2, color='#334155')
    inset.set_yticks(y, labels)
    inset.invert_yaxis()
    inset.set_xlim(-.045, max(relative_deviation[1:])+.13)
    inset.set_xlabel('Relative coefficient error (%)', fontsize=7.3, labelpad=1)
    inset.set_title('Coefficient recovery', fontsize=8.8, pad=2)
    inset.tick_params(axis='both', labelsize=6.8, length=2)
    inset.grid(axis='x', alpha=.18)
    inset.grid(axis='y', visible=False)
    save(fig, 'fig_spider.pdf')


def result_figures(aggregate):
    """Run the standalone manuscript main-results figure script."""
    import runpy

    runpy.run_path(
        str(ROOT / 'paper/figure_code/fig_main_results.py'),
        run_name='__main__',
    )


def training_convergence_figure():
    """Run the standalone manuscript figure script."""
    import runpy

    runpy.run_path(
        str(ROOT / 'paper/figure_code/fig_training_convergence.py'),
        run_name='__main__',
    )


def noise_figure(artifact):
    """Run the standalone manuscript noise-sweep figure script."""
    import runpy

    runpy.run_path(
        str(ROOT / 'paper/figure_code/fig_noise.py'),
        run_name='__main__',
    )


def spider_noise_figure(artifact):
    """Run the standalone manuscript SPIDER noise-sweep figure script."""
    import runpy

    runpy.run_path(
        str(ROOT / 'paper/figure_code/fig_spider_noise.py'),
        run_name='__main__',
    )


def ablation_figure(lambda_artifact, decoder_artifact):
    """Run the standalone manuscript ablation figure script."""
    import runpy

    runpy.run_path(
        str(ROOT / 'paper/figure_code/fig_ablation.py'),
        run_name='__main__',
    )


def latent_figure(artifact):
    """Plot held-out latent projections using the standalone figure script."""
    import runpy

    runpy.run_path(
        str(ROOT / 'paper/figure_code/fig_latent_analysis.py'),
        run_name='__main__',
    )


def efficiency_figure(artifact):
    methods = ('cnn_bp', 'noprop_reference', 'noprop_ct', 'noprop_no_equation',
               'noprop_analytic_ns', 'pi_noprop', 'global_physics_bp',
               'spider_rate_classifier')
    labels = ('CNN', 'NoProp', 'NoProp-CT', 'No equation', 'Analytic NS',
              'PI-NoProp', 'Global physics', 'SPIDER rate')
    x = np.arange(len(methods)); width = .36
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9), constrained_layout=True)
    for ax, key, ylabel, title in ((axes[0], 'train_seconds', 'Training time (s)',
                                    'Measured training time'),
                                   (axes[1], 'peak_memory_mb', 'Peak allocated memory (MB)',
                                    'Measured peak memory')):
        for idx, region in enumerate(('low_enstrophy', 'high_enstrophy')):
            values = [artifact['results'][region][m][key]['mean'] for m in methods]
            errors = [artifact['results'][region][m][key]['std'] for m in methods]
            ax.bar(x+(idx-.5)*width, values, width, yerr=errors, capsize=2.5,
                   color=('#6baed6' if idx == 0 else '#2171b5'),
                   label=('Low enstrophy' if idx == 0 else 'High enstrophy'))
        ax.set_xticks(x, labels, rotation=35, ha='right'); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.grid(axis='y', alpha=.20); ax.grid(axis='x', visible=False)
        ax.legend(frameon=False, fontsize=8)
    save(fig, 'fig_efficiency.pdf')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--protocol', choices=tuple(PROTOCOLS), default='v5',
        help='v5: input32/target16, residual warm-start, lambda=0.1 (default); '
             'old16: explicitly use legacy input16/target16, lambda=0.01',
    )
    return parser


def required_sources(protocol: FigureProtocol = V5) -> tuple[Path, ...]:
    """List dependencies of the cited figures, without legacy fallback."""
    paths = [FIGURE_CODE/f'{name}.py' for name, _ in PAPER_FIGURES]
    paths.extend(protocol.aggregate_path(suffix) for suffix in PAPER_AGGREGATES)
    paths.extend((
        ROOT/'data/generated_hit_ns/manifest.json',
        ROOT/'data/generated_hit_ns/trajectory_000/frame_000.npz',
        ROOT/'outputs/aggregate/full_ns_spider_noise.json',
    ))
    paths.extend(protocol.cache_path/name
                 for name in ('metadata.json', 'fields.npy', 'regions.npy'))
    for region in REGIONS:
        for seed in SEEDS:
            paths.extend(protocol.run_path(region, seed)/name
                         for name in ('history.npz', 'metrics.json', 'config.json'))
    return tuple(paths)


def validate_cache_source(protocol: FigureProtocol = V5) -> None:
    """Read cache metadata and NumPy headers only; never load fields onto a GPU."""
    cache = protocol.cache_path
    metadata = read_json(cache/'metadata.json')
    input_size = metadata.get('input_spatial_size', metadata.get('spatial_size'))
    target_size = metadata.get('target_spatial_size', metadata.get('spatial_size'))
    if (input_size, target_size) != (protocol.input_size, protocol.target_size):
        raise ValueError(f'{cache}: cache geometry does not match {protocol.name}')
    if metadata.get('spatial_size', input_size) != input_size:
        raise ValueError(f'{cache}: spatial_size disagrees with input_spatial_size')
    fields = np.load(cache/'fields.npy', mmap_mode='r', allow_pickle=False)
    regions = np.load(cache/'regions.npy', mmap_mode='r', allow_pickle=False)
    if len(fields.shape) != 5 or fields.shape[1:] != (4,) + (protocol.input_size,)*3:
        raise ValueError(f'{cache}: invalid input fields shape {fields.shape}')
    if regions.shape != (fields.shape[0],):
        raise ValueError(f'{cache}: regions do not match the input samples')
    if metadata.get('n_samples', fields.shape[0]) != fields.shape[0]:
        raise ValueError(f'{cache}: metadata sample count does not match fields')


def preflight_sources(protocol: FigureProtocol = V5) -> None:
    """Fail before rendering any PDF if selected-protocol inputs are missing."""
    missing = [path for path in required_sources(protocol) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f'Missing {protocol.name} figure sources (no legacy fallback):\n'
            + '\n'.join(str(path) for path in missing)
        )
    for suffix in PAPER_AGGREGATES:
        load_aggregate(protocol, suffix)
    validate_cache_source(protocol)
    for region in REGIONS:
        for seed in SEEDS:
            run = protocol.run_path(region, seed)
            metrics = read_json(run/'metrics.json')
            validate_metadata(protocol, metrics, run/'metrics.json')
            if (metrics.get('region'), metrics.get('seed'), metrics.get('physics_source')) != (
                region, seed, 'discovered'):
                raise ValueError(f'{run}: mismatched training-run identity')
            config = read_json(run/'config.json')
            validate_metadata(protocol, {
                'input_spatial_size': config['data']['subdomain_size'],
                'target_spatial_size': config['data'].get('target_subdomain_size')
                                       or config['data']['subdomain_size'],
                'spatial_context_mode': config['noprop'].get('spatial_context_mode', 'single'),
                'lambda_weight': config['physics']['lambda_weight'],
            }, run/'config.json')


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    protocol = PROTOCOLS[args.protocol]
    preflight_sources(protocol)
    # Separate processes keep each figure's tuned rcParams and CLI isolated.
    # A failing child propagates; no historical renderer or source is retried.
    for name, uses_protocol in PAPER_FIGURES:
        command = [sys.executable, '-B', str(FIGURE_CODE/f'{name}.py')]
        if uses_protocol:
            command.extend(('--protocol', protocol.name))
        print(f'Rendering {name} ({protocol.name if uses_protocol else "DNS64"})',
              flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()

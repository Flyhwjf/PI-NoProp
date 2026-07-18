"""Generate manuscript figures directly from the current full-NS artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT/'paper/figures'
FIGURES.mkdir(parents=True, exist_ok=True)
COLORS = {'none': '#8b95a5', 'analytic': '#e99b42', 'discovered': '#2b7bbb',
          'green': '#3a9d72', 'red': '#c44e52', 'navy': '#274c77'}


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': 0.18, 'figure.dpi': 150,
    })


def save(fig, name):
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
    save(fig, 'fig_framework.png')


def local_update():
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    items = [
        (.02, .58, .16, .22, '3-D condition encoder\n+ NS energy rate', '#39739d'),
        (.02, .18, .16, .22, 'Noisy target\nlatent', '#4695b5'),
        (.28, .38, .18, .27, 'Sampled block $J$\nonly optimizer stepped', '#3d9967'),
        (.55, .55, .18, .25, 'Frozen temporal decoder\n$9\\times4\\times16^3$', '#dd762d'),
        (.55, .16, .18, .24, 'Local objective\n$T(\\mathcal{L}_{diff}+\\lambda\\mathcal{L}_{NS})$', '#c74e53'),
        (.81, .55, .17, .25, 'Detached $z_T$\nclassifier', '#39739d'),
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
    ax.text(.895,.28,'No classifier-to-block\nor cross-block gradient',ha='center',
            color=COLORS['red'],weight='bold')
    ax.set_title('Strictly local full-NS block update',fontsize=15,weight='bold')
    save(fig, 'fig_local_training.png')


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
    ax.legend(frameon=False, loc='upper right', fontsize=7.8)
    ax.text(-.12, 1.06, '(a)', transform=ax.transAxes, fontsize=11,
            fontweight='bold')

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
              frameon=False, loc='center left', fontsize=7.8)
    ax.text(.97, .40,
            (f'max div. = {divergence.max():.2e}\n'
             f'max CFL = {cfl.max():.3f} ({100*cfl.max()/.5:.1f}% of limit)'),
            transform=ax.transAxes, ha='right', va='center', fontsize=8.2,
            bbox={'boxstyle': 'round,pad=.35', 'facecolor': 'white',
                  'edgecolor': '#cbd5e1', 'alpha': .92})
    ax.text(-.12, 1.06, '(b)', transform=ax.transAxes, fontsize=11,
            fontweight='bold')
    save(fig, 'fig_dns_quality.png')


def data_samples():
    root = ROOT/'data/cache_hit_ns'
    fields = np.load(root/'fields.npy', mmap_mode='r')
    regions = np.load(root/'regions.npy')
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.5))
    selected_fields = []
    for region in (0, 1):
        index = int(np.flatnonzero(regions == region)[0])
        selected_fields.append(np.asarray(fields[index, 0]))
    limit = max(np.max(np.abs(value)) for value in selected_fields)
    for row, u in enumerate(selected_fields):
        slices = (u[u.shape[0]//2], u[:, u.shape[1]//2], u[:, :, u.shape[2]//2])
        for column, value in enumerate(slices):
            image = axes[row, column].imshow(value.T, origin='lower', cmap='RdBu_r',
                                             vmin=-limit, vmax=limit)
            axes[row, column].set_xticks([]); axes[row, column].set_yticks([])
            if row == 0: axes[row, column].set_title(('x', 'y', 'z')[column]+'-normal')
        axes[row, 0].set_ylabel(('Low' if row == 0 else 'High')+' enstrophy')
    fig.colorbar(image, ax=axes, fraction=.025, pad=.025, label=r'$u_x$')
    fig.suptitle('Trajectory-disjoint decaying-HIT learning samples', y=.98)
    save(fig, 'fig_data_samples.png')


def spider_figure(artifact):
    expected = np.asarray(
        artifact['expected_equation_for_post_discovery_audit']['coefficients'],
        dtype=float)
    discovered = np.asarray(artifact['equation']['coefficients'], dtype=float)
    metrics = artifact['metrics']
    bootstrap_std = np.asarray(metrics['bootstrap_coefficient_std'], dtype=float)
    labels = [r'$\partial_t\mathbf{u}$', r'$(\mathbf{u}\!\cdot\!\nabla)\mathbf{u}$',
              r'$\nabla p$', r'$\nabla^2\mathbf{u}$']
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9),
                             gridspec_kw={'width_ratios': (1.06, 1)},
                             constrained_layout=True)

    # Signed multiplicative deviations are easier to interpret than bars on a
    # truncated coefficient-ratio axis, especially for the negative viscosity.
    relative_deviation = 100*(discovered/expected-1)
    relative_std = 100*np.abs(bootstrap_std/expected)
    y = np.arange(len(labels))
    ax = axes[0]
    ax.axvline(0, color='#64748b', linewidth=1.2, linestyle='--',
               label='Analytic DNS coefficient')
    ax.hlines(y, 0, relative_deviation, color='#bfdbfe', linewidth=5,
              zorder=1)
    ax.errorbar(relative_deviation[1:], y[1:], xerr=relative_std[1:],
                fmt='o', markersize=7, capsize=4, elinewidth=1.5,
                color=COLORS['discovered'], ecolor='#4f94c5', zorder=3,
                label=r'SPIDER $\pm$ one bootstrap s.d.')
    ax.scatter(relative_deviation[0], y[0], marker='D', s=48,
               color=COLORS['none'], zorder=3)
    for index, (deviation, coefficient) in enumerate(
            zip(relative_deviation, discovered)):
        annotation = ('fixed at 1' if index == 0
                      else f'{coefficient:.6f}  ({deviation:+.3f}%)')
        ax.text(deviation+.012, index-.13, annotation, ha='left', va='center',
                fontsize=8, color='#334155')
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(-.045, max(relative_deviation[1:])+.12)
    ax.set_xlabel('Deviation from analytic coefficient (%)')
    ax.set_title('Coefficient recovery and bootstrap stability', pad=13)
    ax.grid(axis='x', alpha=.20); ax.grid(axis='y', visible=False)
    ax.text(.98, .92,
            (f'maximum error: {100*metrics["max_coefficient_relative_error"]:.3f}%\n'
             f'acceptance limit: {100*artifact["config"]["max_coefficient_relative_error"]:.0f}%'),
            transform=ax.transAxes, ha='right', va='top', fontsize=8.2,
            bbox={'boxstyle': 'round,pad=.35', 'facecolor': '#eff6ff',
                  'edgecolor': '#bfdbfe', 'alpha': .95})
    ax.text(-.12, 1.06, '(a)', transform=ax.transAxes, fontsize=11,
            fontweight='bold')

    ax = axes[1]
    residuals = np.asarray([
        metrics['discovery_eta'], metrics['validation_eta'], metrics['test_eta'],
        metrics['next_best_validation_eta']])
    x = np.arange(4)
    threshold = artifact['config']['max_validation_eta']
    ax.axhspan(5e-4, threshold, color='#dcfce7', alpha=.55, zorder=0)
    ax.axhline(threshold, color='#d97706', linestyle='--', linewidth=1.4,
               label=fr'Acceptance threshold $\eta={threshold:g}$')
    ax.plot(x[:3], residuals[:3], color=COLORS['green'], linewidth=2.1,
            marker='o', markersize=7, label='Accepted NS support', zorder=3)
    ax.scatter(x[3], residuals[3], marker='D', s=66, color=COLORS['red'],
               label='Nearest alternative', zorder=3)
    for index, value in enumerate(residuals):
        ax.annotate(f'{value:.2e}', (index, value), xytext=(0, 9),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=7.8, color='#334155')
    ax.annotate(fr'{metrics["support_separation_ratio"]:.1f}$\times$ separation',
                xy=(3, residuals[3]), xytext=(2.15, 8.0e-3),
                arrowprops={'arrowstyle': '->', 'color': COLORS['red'],
                            'linewidth': 1.1},
                fontsize=8.2, color=COLORS['red'], ha='center')
    ax.set_yscale('log'); ax.set_ylim(6e-4, 7e-2)
    ax.set_xticks(x, ['Discovery\n(9 trajectories)',
                      'Validation\n(3 trajectories)',
                      'Test\n(3 trajectories)', 'Nearest\nalternative'])
    ax.set_ylabel(r'Contribution-normalised weak residual $\eta$')
    ax.set_title('Trajectory-disjoint support validation', pad=13)
    ax.grid(axis='y', which='both', alpha=.20); ax.grid(axis='x', visible=False)
    ax.legend(frameon=False, loc='upper left', fontsize=7.7)
    ax.text(.97, .30, '100/100 bootstrap refits\nretain all four NS terms',
            transform=ax.transAxes, ha='right', va='center', fontsize=8.2,
            bbox={'boxstyle': 'round,pad=.35', 'facecolor': '#f0fdf4',
                  'edgecolor': '#bbf7d0', 'alpha': .95})
    ax.text(-.12, 1.06, '(b)', transform=ax.transAxes, fontsize=11,
            fontweight='bold')
    save(fig, 'fig_spider.png')


def result_figures(aggregate):
    regions = ('low_enstrophy', 'high_enstrophy')
    methods = ('none', 'analytic', 'discovered')
    labels = ('No physics', 'Analytic NS', 'Discovered NS')
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.65),
                             gridspec_kw={'width_ratios': (1.16, 1, 1)},
                             constrained_layout=True)
    x = np.arange(3)
    region_styles = (
        ('low_enstrophy', '#6baed6', 'o', 'Low enstrophy'),
        ('high_enstrophy', '#2171b5', 's', 'High enstrophy'))
    for axis, metric, title in zip(
            axes, ('accuracy', 'eta_ns', 'eta_div'),
            ('Future-decay classification', 'Full-NS consistency',
             'Continuity consistency')):
        axis.axvspan(1.67, 2.33, color='#dcfce7', alpha=.48, zorder=0)
        for region, colour, marker, label in region_styles:
            means = np.asarray([aggregate['results'][region][method][metric]['mean']
                                for method in methods])
            stds = np.asarray([aggregate['results'][region][method][metric]['std']
                               for method in methods])
            axis.errorbar(x, means, yerr=stds, color=colour, marker=marker,
                          markersize=6.5, linewidth=1.8, capsize=4,
                          label=label, zorder=3)
        axis.set_xticks(x, labels, rotation=15)
        axis.set_title(title, pad=11)
        axis.grid(axis='y', alpha=.20); axis.grid(axis='x', visible=False)
    axes[0].axhline(20, color='#64748b', linestyle=':', linewidth=1.2,
                    label='Five-class chance')
    axes[0].set_ylabel('Test accuracy (%)')
    axes[0].set_ylim(8, 94)
    axes[0].annotate('+69.14 pp', xy=(2, 86.73), xytext=(.38, 69),
                     arrowprops={'arrowstyle': '->', 'color': '#3a9d72',
                                 'linewidth': 1.1},
                     color='#287a56', fontsize=8, ha='center')
    axes[0].annotate('+65.48 pp', xy=(2, 82.94), xytext=(1.05, 57),
                     arrowprops={'arrowstyle': '->', 'color': '#2171b5',
                                 'linewidth': 1.1},
                     color='#175a91', fontsize=8, ha='center')
    axes[0].legend(frameon=False, fontsize=7.7, loc='lower right')
    axes[1].set_ylabel(r'$\eta_{\mathrm{NS}}$ (lower is better)')
    axes[1].set_ylim(0, 1.08)
    axes[2].set_ylabel(r'$\eta_{\mathrm{div}}$ (lower is better)')
    axes[2].set_ylim(0, .63)
    for index, axis in enumerate(axes):
        axis.text(-.13, 1.06, f'({chr(97+index)})', transform=axis.transAxes,
                  fontsize=10.5, fontweight='bold')
    save(fig, 'fig_main_results.png')

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.3))
    for axis, key, title, ylabel in (
        (axes[0], 'block_seconds', 'Local block time', 'Seconds'),
        (axes[1], 'peak_memory_mb', 'Peak allocated memory', 'MB')):
        x=np.arange(3); width=.34
        for ridx, region in enumerate(regions):
            values=[aggregate['results'][region][m][key]['mean'] for m in methods]
            axis.bar(x+(ridx-.5)*width, values, width,
                     color=('#6baed6' if ridx == 0 else '#2171b5'),
                     label=('Low' if ridx == 0 else 'High'))
        axis.set_xticks(x, labels, rotation=18); axis.set_title(title); axis.set_ylabel(ylabel)
    axes[0].legend(frameon=False); fig.tight_layout(); save(fig, 'fig_efficiency.png')


def training_convergence_figure():
    """Aggregate current v4 histories without retraining or test-set tuning."""
    regions = ('low_enstrophy', 'high_enstrophy')
    seeds = (42, 123, 456)
    local_curves = {key: [] for key in ('loss', 'diff', 'phys')}
    classifier = {
        region: {'train': [], 'val': []} for region in regions}
    for region in regions:
        for seed in seeds:
            path = (ROOT/'outputs/runs'/
                    f'full_ns_v4_discovered_{region}_lambda0p01_seed{seed}'/
                    'history.npz')
            if not path.exists():
                raise FileNotFoundError(f'Missing current v4 history: {path}')
            with np.load(path, allow_pickle=True) as history:
                local = history['local'].tolist()
                classifier_history = history['classifier'].tolist()
            if len(local) != 1000 or len(classifier_history) != 30:
                raise ValueError(f'Incomplete v4 history: {path}')
            for block in range(10):
                sequence = [record for record in local if record['block'] == block]
                if len(sequence) != 100:
                    raise ValueError(f'Block {block} is incomplete in {path}')
                for key in local_curves:
                    values = np.asarray([record[key] for record in sequence], dtype=float)
                    reference = np.median(values[:10])
                    local_curves[key].append(values/max(reference, 1e-12))
            classifier[region]['train'].append(
                [record['train']['accuracy'] for record in classifier_history])
            classifier[region]['val'].append(
                [record['val']['accuracy'] for record in classifier_history])

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9),
                             gridspec_kw={'width_ratios': (1.04, 1)},
                             constrained_layout=True)
    ax = axes[0]
    updates = np.arange(1, 101)
    curve_styles = (
        ('loss', COLORS['navy'], 'Total local objective'),
        ('diff', COLORS['discovered'], 'Diffusion term'),
        ('phys', COLORS['green'], 'Physics term'))
    for key, colour, label in curve_styles:
        values = np.asarray(local_curves[key])
        median = np.median(values, axis=0)
        lower, upper = np.quantile(values, [.25, .75], axis=0)
        ax.fill_between(updates, lower, upper, color=colour, alpha=.14,
                        linewidth=0)
        ax.plot(updates, median, color=colour, linewidth=2, label=label)
    ax.set_yscale('log')
    ax.set(xlabel='Local updates per block',
           ylabel='Normalised loss (first 10-update median = 1)',
           title='Strictly local block optimisation')
    ax.grid(axis='y', which='both', alpha=.20); ax.grid(axis='x', alpha=.12)
    ax.legend(frameon=False, fontsize=7.8, loc='lower left')
    ax.text(.97, .94, 'median and interquartile range\n60 block--seed--region curves',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            bbox={'boxstyle': 'round,pad=.35', 'facecolor': 'white',
                  'edgecolor': '#cbd5e1', 'alpha': .93})
    ax.text(-.12, 1.06, '(a)', transform=ax.transAxes, fontsize=11,
            fontweight='bold')

    ax = axes[1]
    epochs = np.arange(1, 31)
    for region, colour, label in (
            ('low_enstrophy', '#6baed6', 'Low enstrophy'),
            ('high_enstrophy', '#2171b5', 'High enstrophy')):
        train = np.asarray(classifier[region]['train'])
        validation = np.asarray(classifier[region]['val'])
        ax.plot(epochs, train.mean(axis=0), color=colour, linestyle='--',
                linewidth=1.25, alpha=.72, label=f'{label}, train')
        ax.plot(epochs, validation.mean(axis=0), color=colour, linewidth=2.1,
                label=f'{label}, validation')
        validation_std = validation.std(axis=0, ddof=1)
        ax.fill_between(epochs, validation.mean(axis=0)-validation_std,
                        validation.mean(axis=0)+validation_std,
                        color=colour, alpha=.14, linewidth=0)
    ax.axhline(20, color='#64748b', linestyle=':', linewidth=1.2,
               label='Five-class chance')
    ax.set(xlabel='Detached-classifier epoch', ylabel='Accuracy (%)',
           ylim=(10, 101), title='Classifier convergence after local training')
    ax.grid(axis='y', alpha=.20); ax.grid(axis='x', alpha=.12)
    ax.legend(frameon=False, fontsize=7.3, ncol=2, loc='lower right')
    ax.text(-.12, 1.06, '(b)', transform=ax.transAxes, fontsize=11,
            fontweight='bold')
    save(fig, 'fig_training_convergence.png')


def noise_figure(artifact):
    levels = np.asarray(artifact['protocol']['levels_in_channel_standard_deviations'])
    positions = np.arange(len(levels))
    tick_labels = [f'{100*level:g}%' for level in levels]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9),
                             gridspec_kw={'width_ratios': (1.04, 1)},
                             constrained_layout=True)
    for axis in axes:
        axis.axvspan(-.35, 2.35, color='#dcfce7', alpha=.48, zorder=0)
        axis.axvspan(2.35, 4.35, color='#fef3c7', alpha=.48, zorder=0)
        axis.axvspan(4.35, 6.35, color='#fee2e2', alpha=.40, zorder=0)
    for region, colour, marker, region_label in (
            ('low_enstrophy', '#6baed6', 'o', 'Low enstrophy'),
            ('high_enstrophy', '#2171b5', 's', 'High enstrophy')):
        for method, linestyle, method_label, alpha in (
                ('none', '--', 'No equation', .62),
                ('discovered', '-', 'PI-NoProp', 1.0)):
            records = artifact['results'][region][method]
            accuracy = np.asarray([
                records[str(float(level))]['accuracy']['mean'] for level in levels])
            accuracy_std = np.asarray([
                records[str(float(level))]['accuracy']['std'] for level in levels])
            eta = np.asarray([
                records[str(float(level))]['eta_ns']['mean'] for level in levels])
            eta_std = np.asarray([
                records[str(float(level))]['eta_ns']['std'] for level in levels])
            label = f'{method_label}, {region_label.lower()}'
            axes[0].plot(positions, accuracy, marker=marker, linestyle=linestyle,
                         color=colour, alpha=alpha, linewidth=2 if method == 'discovered'
                         else 1.35, markersize=6, label=label)
            axes[0].fill_between(positions, accuracy-accuracy_std,
                                 accuracy+accuracy_std, color=colour,
                                 alpha=.12 if method == 'discovered' else .06,
                                 linewidth=0)
            axes[1].plot(positions, eta, marker=marker, linestyle=linestyle,
                         color=colour, alpha=alpha, linewidth=2 if method == 'discovered'
                         else 1.35, markersize=6, label=label)
            axes[1].fill_between(positions, np.maximum(eta-eta_std, 0),
                                 eta+eta_std, color=colour,
                                 alpha=.12 if method == 'discovered' else .06,
                                 linewidth=0)
    axes[0].axhline(20, color='#64748b', linestyle=':', linewidth=1.25)
    axes[0].text(6.05, 20.8, 'chance', color='#475569', fontsize=7.8,
                 ha='right', va='bottom')
    axes[0].set(xlabel='Observation noise / channel standard deviation',
                ylabel='Test accuracy (%)', ylim=(10, 94),
                title='Clean-trained predictive robustness')
    axes[1].set(xlabel='Observation noise / channel standard deviation',
                ylabel=r'$\eta_{\mathrm{NS}}$ (lower is better)', ylim=(0, 1.02),
                title='Decoded-field equation residual')
    for axis in axes:
        axis.set_xticks(positions, tick_labels)
        axis.grid(axis='y', alpha=.20); axis.grid(axis='x', visible=False)
        axis.text(.13, .96, 'stable', transform=axis.transAxes, ha='center',
                  va='top', fontsize=7.8, color='#287a56', fontweight='bold')
        axis.text(.50, .96, 'degradation', transform=axis.transAxes, ha='center',
                  va='top', fontsize=7.8, color='#9a6700', fontweight='bold')
        axis.text(.84, .96, 'near chance', transform=axis.transAxes, ha='center',
                  va='top', fontsize=7.8, color='#a33b3b', fontweight='bold')
    axes[0].legend(frameon=False, fontsize=7.25, ncol=1,
                   loc='center left', bbox_to_anchor=(.02, .42))
    axes[1].annotate('Residual decreases while\nclassification collapses',
                     xy=(6, artifact['results']['high_enstrophy']['discovered']['1.0']
                        ['eta_ns']['mean']), xytext=(4.0, .42),
                     arrowprops={'arrowstyle': '->', 'color': '#c44e52',
                                 'linewidth': 1.1},
                     fontsize=8, color='#a33b3b', ha='center')
    for index, axis in enumerate(axes):
        axis.text(-.12, 1.06, f'({chr(97+index)})', transform=axis.transAxes,
                  fontsize=11, fontweight='bold')
    save(fig, 'fig_noise.png')


def spider_noise_figure(artifact):
    levels = np.asarray(artifact['levels_in_field_standard_deviations'], dtype=float)
    positions = np.arange(len(levels))
    records = [artifact['levels'][str(float(level))] for level in levels]
    tick_labels = [f'{100*level:g}%' for level in levels]
    coefficient_error = 100*np.asarray(
        [record['max_coefficient_relative_error'] for record in records])
    bootstrap = 100*np.asarray(
        [record['bootstrap_support_fraction'] for record in records])
    discovery_eta = np.asarray([record['discovery_eta'] for record in records])
    validation_eta = np.asarray([record['validation_eta'] for record in records])
    test_eta = np.asarray([record['test_eta'] for record in records])
    passed = np.asarray(
        [record['validation_passed_under_noise_protocol'] for record in records])
    protocol = artifact['protocol']

    fig, axes = plt.subplots(1, 2, figsize=(10.9, 3.9),
                             constrained_layout=True)
    for axis in axes:
        axis.axvspan(-.35, 2.35, color='#dcfce7', alpha=.52, zorder=0)
        axis.axvspan(2.35, len(levels)-.65, color='#fee2e2', alpha=.34, zorder=0)
        axis.axvline(2.5, color='#c44e52', linestyle=':', linewidth=1.2)

    ax = axes[0]
    coefficient_line, = ax.plot(
        positions, coefficient_error, color=COLORS['discovered'], marker='o',
        linewidth=2, markersize=6, label='Maximum coefficient error')
    ax.set_yscale('log'); ax.set_ylim(.01, 1600)
    coefficient_limit = 100*protocol['max_coefficient_relative_error']
    ax.axhline(coefficient_limit, color='#d97706', linestyle='--', linewidth=1.3)
    ax.set_ylabel('Maximum coefficient error (%)', color='#175a91')
    ax.tick_params(axis='y', colors='#175a91')
    ax2 = ax.twinx()
    bootstrap_line, = ax2.plot(
        positions, bootstrap, color=COLORS['green'], marker='s', linewidth=1.8,
        markersize=5.5, label='Bootstrap support')
    bootstrap_limit = 100*protocol['min_bootstrap_support']
    ax2.axhline(bootstrap_limit, color='#3a9d72', linestyle='--', linewidth=1.2)
    ax2.set_ylim(0, 108); ax2.set_ylabel('Selected-support bootstrap rate (%)',
                                         color='#287a56')
    ax2.tick_params(axis='y', colors='#287a56')
    ax2.spines['right'].set_visible(True)
    ax.set_title('Coefficient and support stability', pad=12)
    ax.legend([coefficient_line, bootstrap_line],
              ['Maximum coefficient error', 'Selected-support bootstrap'],
              frameon=False, fontsize=7.6, loc='lower left')
    ax.text(8.0, coefficient_limit*1.08, f'{coefficient_limit:g}% error limit',
            color='#9a6700', fontsize=7.3, ha='right', va='bottom')
    ax2.text(8.0, bootstrap_limit+2, f'{bootstrap_limit:g}% support limit',
             color='#287a56', fontsize=7.3, ha='right', va='bottom')
    ax.annotate('first support failure:\nextra kinetic-energy-gradient term',
                xy=(3, coefficient_error[3]), xytext=(4.7, .12),
                arrowprops={'arrowstyle': '->', 'color': COLORS['red'],
                            'linewidth': 1.1},
                fontsize=7.8, color='#a33b3b', ha='center')

    ax = axes[1]
    ax.plot(positions, discovery_eta, color='#94a3b8', marker='o', linewidth=1.5,
            label='Discovery')
    ax.plot(positions, validation_eta, color=COLORS['discovered'], marker='s',
            linewidth=2, label='Validation')
    ax.plot(positions, test_eta, color=COLORS['navy'], marker='^', linewidth=2,
            label='Held-out test')
    residual_limit = protocol['max_validation_eta']
    ax.axhline(residual_limit, color='#d97706', linestyle='--', linewidth=1.3,
               label=fr'Residual limit $\eta={residual_limit:g}$')
    ax.set_yscale('log'); ax.set_ylim(2e-5, 1.4)
    for index, is_passed in enumerate(passed):
        ax.scatter(index, 4e-5, marker='o' if is_passed else 'x',
                   s=42, linewidth=1.6,
                   color=COLORS['green'] if is_passed else COLORS['red'],
                   zorder=4)
    ax.text(.02, .06, '● accepted', transform=ax.transAxes, color='#287a56',
            fontsize=8)
    ax.text(.23, .06, '× rejected', transform=ax.transAxes, color='#a33b3b',
            fontsize=8)
    ax.set_title('Weak-residual validation under discovery noise', pad=12)
    ax.set_ylabel(r'Weak residual $\eta$')
    ax.legend(frameon=False, fontsize=7.4, loc='upper left', ncol=2)
    for axis in axes:
        axis.set_xticks(positions, tick_labels)
        axis.set_xlabel('DNS-field noise / field standard deviation')
        axis.grid(axis='y', which='both', alpha=.20); axis.grid(axis='x', visible=False)
        if axis is axes[0]:
            axis.text(.14, .80, 'validated', transform=axis.transAxes,
                      ha='center', va='top', fontsize=7.8, color='#287a56',
                      fontweight='bold')
            axis.text(.68, .96, 'rejected', transform=axis.transAxes,
                      ha='center', va='top', fontsize=7.8, color='#a33b3b',
                      fontweight='bold')
    for index, axis in enumerate(axes):
        axis.text(-.12, 1.06, f'({chr(97+index)})', transform=axis.transAxes,
                  fontsize=11, fontweight='bold')
    save(fig, 'fig_spider_noise.png')


def ablation_figure(lambda_artifact, decoder_artifact):
    """Plot only metrics from the reproducible 4.5 ablation artifacts."""
    lambdas = np.asarray(lambda_artifact['protocol']['lambdas'], dtype=float)
    x = np.arange(len(lambdas))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8),
                             gridspec_kw={'width_ratios': (1.04, 1)},
                             constrained_layout=True)
    ax = axes[0]
    for key, colour, label in (
            ('accuracy', COLORS['navy'], 'Accuracy'),
            ('eta_ns', COLORS['discovered'], r'$η_{\mathrm{NS}}$'),
            ('eta_div', COLORS['green'], r'$η_{\mathrm{div}}$')):
        means = np.asarray([lambda_artifact['results'][str(float(v))][key]['mean']
                            for v in lambdas])
        stds = np.asarray([lambda_artifact['results'][str(float(v))][key]['std']
                           for v in lambdas])
        if key == 'accuracy':
            means = means / 100; stds = stds / 100
        ax.errorbar(x, means, yerr=stds, marker='o', linewidth=1.8,
                    markersize=5.5, capsize=3, color=colour, label=label)
    ax.axhline(.2, color='#64748b', linestyle=':', linewidth=1.1)
    ax.set_xticks(x, [f'{v:g}' for v in lambdas])
    ax.set_xlabel(r'Physics-loss weight $λ$')
    ax.set_ylabel('Accuracy / normalised residual')
    ax.set_ylim(0, 1.03); ax.set_title('Weight trade-off (low enstrophy)')
    ax.grid(axis='y', alpha=.20); ax.grid(axis='x', visible=False)
    ax.legend(frameon=False, fontsize=8)
    ax.text(-.12, 1.06, '(a)', transform=ax.transAxes, fontsize=11,
            fontweight='bold')

    ax = axes[1]
    architectures = lambda_artifact.get('protocol', {}).get('architectures',
                                                              ['linear', 'conv'])
    architectures = decoder_artifact['protocol']['architectures']
    labels = ['Linear', 'Convolutional']
    metrics = ('accuracy', 'eta_ns', 'eta_div')
    metric_labels = ('Accuracy', r'$η_{\mathrm{NS}}$', r'$η_{\mathrm{div}}$')
    xpos = np.arange(len(metrics)); width = .32
    for index, architecture in enumerate(architectures):
        values = []
        errors = []
        for key in metrics:
            value = decoder_artifact['results'][architecture][key]
            mean, std = value['mean'], value['std']
            if key == 'accuracy': mean, std = mean/100, std/100
            values.append(mean); errors.append(std)
        ax.bar(xpos+(index-.5)*width, values, width, yerr=errors,
               capsize=3, color=('#94a3b8' if architecture == 'linear'
                                 else COLORS['discovered']),
               label=labels[index])
    ax.set_xticks(xpos, metric_labels)
    ax.set_ylabel('Accuracy / normalised residual')
    ax.set_ylim(0, 1.03); ax.set_title('Decoder architecture (λ=0.01)')
    ax.grid(axis='y', alpha=.20); ax.grid(axis='x', visible=False)
    ax.legend(frameon=False, fontsize=8)
    ax.text(-.12, 1.06, '(b)', transform=ax.transAxes, fontsize=11,
            fontweight='bold')
    save(fig, 'fig_ablation.png')


def latent_figure(artifact):
    """Plot held-out latent projections and the corresponding metrics."""
    method_colours = {'none': '#8b95a5', 'analytic': '#e99b42',
                      'discovered': COLORS['discovered']}
    method_labels = {'none': 'No equation', 'analytic': 'Analytic NS',
                     'discovered': 'PI-NoProp'}
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.0), constrained_layout=True)
    for row, projection in enumerate(('pca', 'tsne')):
        for col, region in enumerate(('low_enstrophy', 'high_enstrophy')):
            ax = axes[row, col]
            for method in ('none', 'analytic', 'discovered'):
                points = artifact['points'][region][method]
                xy = np.asarray(points[projection]); labels = np.asarray(points['labels'])
                for cls in np.unique(labels):
                    mask = labels == cls
                    ax.scatter(xy[mask, 0], xy[mask, 1], s=10, alpha=.50,
                               color=method_colours[method], marker=('o','s','^')[int(cls)%3],
                               label=method_labels[method] if cls == np.unique(labels)[0] else None)
            ax.set_title(('PCA' if projection == 'pca' else 't-SNE') +
                         f' — {"Low" if col == 0 else "High"} enstrophy')
            ax.set_xlabel(f'{projection.upper()} 1'); ax.set_ylabel(f'{projection.upper()} 2')
            ax.grid(alpha=.16)
            if row == 0 and col == 0: ax.legend(frameon=False, fontsize=8)
    save(fig, 'fig_latent_analysis.png')


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
    save(fig, 'fig_efficiency.png')


def main():
    style(); framework(); local_update()
    manifest = json.loads((ROOT/'data/generated_hit_ns/manifest.json')
                          .read_text(encoding='utf-8'))
    artifact = json.loads((ROOT/'outputs/spider/full_ns_equation.json')
                          .read_text(encoding='utf-8'))
    dns_quality(manifest); data_samples(); spider_figure(artifact)
    aggregate_path = ROOT/'outputs/aggregate/full_ns_results.json'
    if aggregate_path.exists():
        result_figures(json.loads(aggregate_path.read_text(encoding='utf-8')))
        training_convergence_figure()
    else:
        print('Aggregate not present; discovery figures generated, result figures deferred.')
    noise_path = ROOT/'outputs/aggregate/full_ns_noise.json'
    if noise_path.exists():
        noise_figure(json.loads(noise_path.read_text(encoding='utf-8')))
    spider_noise_path = ROOT/'outputs/aggregate/full_ns_spider_noise.json'
    if spider_noise_path.exists():
        spider_noise_figure(json.loads(
            spider_noise_path.read_text(encoding='utf-8')))
    lambda_path = ROOT/'outputs/aggregate/full_ns_lambda_ablation.json'
    decoder_path = ROOT/'outputs/aggregate/full_ns_decoder_ablation.json'
    if lambda_path.exists() and decoder_path.exists():
        ablation_figure(json.loads(lambda_path.read_text(encoding='utf-8')),
                        json.loads(decoder_path.read_text(encoding='utf-8')))
    latent_path = ROOT/'outputs/aggregate/full_ns_latent_analysis.json'
    baseline_path = ROOT/'outputs/aggregate/full_ns_baselines.json'
    if latent_path.exists():
        latent_figure(json.loads(latent_path.read_text(encoding='utf-8')))
    if baseline_path.exists():
        efficiency_figure(json.loads(baseline_path.read_text(encoding='utf-8')))


if __name__ == '__main__':
    main()

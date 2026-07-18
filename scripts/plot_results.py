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
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
    for record in manifest['trajectories']:
        d = record['diagnostics']; t = np.asarray([x['time'] for x in d])
        e = np.asarray([x['kinetic_energy'] for x in d])
        axes[0].plot(t-t[0], e/e[0], alpha=.55, lw=1)
    axes[0].set(xlabel='Window time', ylabel=r'$K(t)/K(0)$', title='Monotone decay')
    div = [r['quality']['max_divergence_rms'] for r in manifest['trajectories']]
    cfl = [r['quality']['max_cfl'] for r in manifest['trajectories']]
    axes[1].bar(np.arange(15), div, color=COLORS['green'])
    axes[1].set(xlabel='Trajectory', ylabel='Maximum divergence RMS',
                title='Incompressibility')
    axes[1].set_yscale('log')
    axes[2].bar(np.arange(15), cfl, color=COLORS['navy'])
    axes[2].axhline(.5, ls='--', color=COLORS['red'], label='quality limit')
    axes[2].set(xlabel='Trajectory', ylabel='Maximum CFL', title='Temporal stability')
    axes[2].legend(frameon=False)
    fig.tight_layout(); save(fig, 'fig_dns_quality.png')


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
    expected = artifact['expected_equation_for_post_discovery_audit']['coefficients']
    discovered = artifact['equation']['coefficients']
    labels = [r'$\partial_tu$', r'$(u\cdot\nabla)u$', r'$\nabla p$', r'$\nabla^2u$']
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    x = np.arange(4); width=.36
    ratios = np.asarray(discovered)/np.asarray(expected)
    axes[0].bar(x-width/2, np.ones(4), width, label='DNS truth', color=COLORS['none'])
    bars = axes[0].bar(x+width/2, ratios, width, label='SPIDER',
                       color=COLORS['discovered'])
    for bar, value in zip(bars, discovered):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+.003,
                     f'{value:.5f}', ha='center', va='bottom', fontsize=7)
    axes[0].set_ylim(0.97, 1.025)
    axes[0].set_xticks(x, labels); axes[0].set_ylabel('Coefficient / DNS truth')
    axes[0].set_title('Recovered full momentum equation'); axes[0].legend(frameon=False)
    m = artifact['metrics']
    values = [m['discovery_eta'], m['validation_eta'], m['test_eta'],
              m['next_best_validation_eta']]
    axes[1].bar(np.arange(4), values,
                color=[COLORS['green']]*3+[COLORS['red']])
    axes[1].set_xticks(np.arange(4), ['Discovery', 'Validation', 'Test', 'Next support'],
                       rotation=15)
    axes[1].set_ylabel(r'Weak residual $\eta$'); axes[1].set_yscale('log')
    axes[1].set_title('Independent support validation')
    fig.tight_layout(); save(fig, 'fig_spider.png')


def result_figures(aggregate):
    regions = ('low_enstrophy', 'high_enstrophy')
    methods = ('none', 'analytic', 'discovered')
    labels = ('No physics', 'Analytic NS', 'Discovered NS')
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for axis, metric, title in zip(
            axes, ('accuracy', 'eta_ns', 'eta_div'),
            ('Future-decay accuracy', 'Full-NS consistency', 'Continuity consistency')):
        x=np.arange(3); width=.34
        for ridx, region in enumerate(regions):
            means=[aggregate['results'][region][m][metric]['mean'] for m in methods]
            stds=[aggregate['results'][region][m][metric]['std'] for m in methods]
            axis.bar(x+(ridx-.5)*width, means, width, yerr=stds, capsize=3,
                     label=('Low' if ridx == 0 else 'High')+' enstrophy',
                     color=('#6baed6' if ridx == 0 else '#2171b5'))
        axis.set_xticks(x, labels, rotation=18); axis.set_title(title)
        axis.set_ylabel('Accuracy (%)' if metric == 'accuracy' else r'$\eta$ (lower is better)')
    axes[0].legend(frameon=False); fig.tight_layout()
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


def noise_figure(artifact):
    levels = np.asarray(artifact['protocol']['levels_in_channel_standard_deviations'])
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3))
    for region, color, marker in (
            ('low_enstrophy', '#6baed6', 'o'),
            ('high_enstrophy', '#2171b5', 's')):
        for method, linestyle, label_method in (
                ('none', '--', 'Vanilla'), ('discovered', '-', 'Discovered PI')):
            records = artifact['results'][region][method]
            accuracy = [records[str(float(level))]['accuracy']['mean'] for level in levels]
            eta = [records[str(float(level))]['eta_ns']['mean'] for level in levels]
            label = f'{label_method}, {"low" if region.startswith("low") else "high"}'
            axes[0].plot(levels, accuracy, marker=marker, linestyle=linestyle,
                         color=color, label=label)
            axes[1].plot(levels, eta, marker=marker, linestyle=linestyle,
                         color=color, label=label)
    axes[0].axhline(20, color='#777777', ls=':', label='chance')
    axes[0].set(xlabel='Gaussian noise / channel std.', ylabel='Accuracy (%)',
                title='Clean-trained prediction')
    axes[1].set(xlabel='Gaussian noise / channel std.', ylabel=r'$\eta_{NS}$',
                title='Decoded-field consistency')
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout(); save(fig, 'fig_noise.png')


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
    else:
        print('Aggregate not present; discovery figures generated, result figures deferred.')
    noise_path = ROOT/'outputs/aggregate/full_ns_noise.json'
    if noise_path.exists():
        noise_figure(json.loads(noise_path.read_text(encoding='utf-8')))


if __name__ == '__main__':
    main()

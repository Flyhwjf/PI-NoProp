"""Paper figures for validated SPIDER discovery and the matched lambda sweep."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    out = Path('paper/figures/legacy')
    out.mkdir(parents=True, exist_ok=True)
    artifact = json.loads(Path('outputs/spider/hit_spatial_equation.json').read_text())
    results = json.loads(Path('outputs/aggregate/discovery_results.json').read_text())

    terms = artifact['equation']['terms']
    coefficients = artifact['equation']['coefficients']
    metrics = artifact['metrics']
    labels = {'pressure_laplacian': r'$\nabla^2p$',
              'convection_divergence': r'$\nabla\!\cdot[(u\!\cdot\!\nabla)u]$'}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), constrained_layout=True)
    axes[0].bar([labels[item] for item in terms], coefficients,
                color=['#2864a8', '#df7b2f'])
    axes[0].axhline(0, color='black', linewidth=0.8)
    axes[0].set_ylabel('Discovered coefficient')
    axes[0].set_title('Validated SPIDER relation')
    values = [metrics['validation_eta'], metrics['next_best_validation_eta']]
    axes[1].bar(['selected support', 'next-best support'], values,
                color=['#3a9d5d', '#999999'])
    axes[1].set_ylabel('Held-out normalized residual')
    axes[1].set_title('Held-out validation')
    axes[1].text(0, values[0] + 0.02, '100% bootstrap\nsupport',
                 ha='center', va='bottom', fontsize=9)
    fig.savefig(out / 'fig_spider_discovery.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    sweep = results['lambda_sweep']
    lambdas = np.asarray([float(item) for item in sweep])
    order = np.argsort(lambdas)
    lambdas = lambdas[order]
    records = [sweep[str(value)] if str(value) in sweep
               else sweep[f'{value:g}'] for value in lambdas]
    accuracy = np.asarray([item['accuracy_mean'] for item in records])
    accuracy_std = np.asarray([item['accuracy_std'] for item in records])
    eta_div = np.asarray([item['eta_div_mean'] for item in records])
    eta_sp = np.asarray([item['eta_pp_discovered_mean'] for item in records])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), constrained_layout=True)
    axes[0].errorbar(lambdas, accuracy, yerr=accuracy_std, marker='o', capsize=3,
                     color='#2864a8')
    axes[0].set_xscale('log')
    axes[0].set_xlabel(r'$\lambda$')
    axes[0].set_ylabel('Test accuracy (%)')
    axes[0].set_title('Three-seed classification')
    axes[1].plot(lambdas, eta_div, marker='s', label=r'$\eta_{div}$')
    axes[1].plot(lambdas, eta_sp, marker='o', label=r'$\eta_{SPIDER}$')
    axes[1].set_xscale('log')
    axes[1].set_xlabel(r'$\lambda$')
    axes[1].set_ylabel('Normalized weak residual')
    axes[1].set_title('Physical consistency')
    axes[1].legend(frameon=False)
    fig.savefig(out / 'fig_lambda_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(out / 'fig_spider_discovery.png')
    print(out / 'fig_lambda_tradeoff.png')


if __name__ == '__main__':
    main()

"""regenerate_figures.py — Generate all paper figures from saved data files.
Read-only script: no training, just reads .npz/.json and generates PNGs.
"""
import sys, os, json, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization import (
    set_global_style,
    plot_data_samples,
    plot_training_curves,
    plot_field_reconstruction_from_data,
    plot_per_class_with_confusion,
    plot_tsne,
    plot_spider_coefficients,
    plot_noise_robustness,
    plot_lambda_tradeoff,
)

set_global_style()
OUT = Path('paper/figures')
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path('data/generated')
OUTPUTS = Path('outputs')

warnings.filterwarnings('ignore')


def fig_data_samples():
    if (DATA / 'centre').exists():
        plot_data_samples(str(DATA), n_slices=3, save_path=str(OUT / 'fig_data_samples.png'))
        print('[OK] fig_data_samples.png')
    else:
        print('[SKIP] No generated data')


def fig_spider_coeffs():
    for fname, title in [('vector_equation.json', 'Momentum Equation — SPIDER Coefficients'),
                          ('scalar_equation.json', 'Continuity Equation — SPIDER Coefficients')]:
        path = OUTPUTS / 'spider' / fname
        if path.exists():
            with open(path) as f:
                eq = json.load(f)
            coeffs = np.array(eq['coefficients'])
            terms = eq['terms']
            # Map to display labels
            display = [t.replace('_', ' ').replace('deriv', '').replace('field', '')
                       for t in terms]
            out_name = 'fig_spider_vector.png' if 'vector' in fname else 'fig_spider_scalar.png'
            plot_spider_coefficients(coeffs, display, title, str(OUT / out_name))
            print(f'[OK] {out_name}')
        else:
            print(f'[SKIP] No SPIDER data: {fname}')


def fig_training_curves():
    path = OUTPUTS / 'data' / 'training_curves' / 'pi_noprop_centre.npz'
    if path.exists():
        raw = dict(np.load(path, allow_pickle=True))
        # Map keys to what plot_training_curves expects
        history = {
            'train_loss': raw.get('loss', raw.get('train_loss', [])),
            'val_loss': raw.get('loss', raw.get('val_loss', [])),
            'val_acc': raw.get('acc', raw.get('val_acc', [])),
            'cls_loss': raw.get('cls', raw.get('cls_loss', [])),
            'diff_loss': raw.get('diff', raw.get('diff_loss', [])),
            'phys_loss': raw.get('phys', raw.get('phys_loss', [])),
        }
        plot_training_curves(history, 'PI-NoProp (centre)', str(OUT / 'fig_training_curves.png'))
        print('[OK] fig_training_curves.png')
    else:
        print('[SKIP] No training curves data')


def fig_field_reconstruction():
    for region in ['centre', 'edge']:
        path = OUTPUTS / 'data' / 'reconstruction' / f'{region}.npz'
        if path.exists():
            data = np.load(path, allow_pickle=True)
            plot_field_reconstruction_from_data(
                data['fields_true'][:4], data['fields_pred'][:4],
                n_samples=3, save_path=str(OUT / f'fig_field_reconstruction_{region}.png'))
            print(f'[OK] fig_field_reconstruction_{region}.png')
        else:
            print(f'[SKIP] No reconstruction data for {region}')

    # Combined figure: use centre data
    path = OUTPUTS / 'data' / 'reconstruction' / 'centre.npz'
    if path.exists():
        data = np.load(path, allow_pickle=True)
        plot_field_reconstruction_from_data(
            data['fields_true'][:4], data['fields_pred'][:4],
            n_samples=3, save_path=str(OUT / 'fig_field_reconstruction.png'))


def fig_region_comparison():
    path = OUTPUTS / 'data' / 'confusion' / 'pi_noprop_centre.npz'
    if path.exists():
        data = np.load(path)
        plot_per_class_with_confusion(
            data['preds'], data['labels'], 10,
            str(OUT / 'fig_region_comparison.png'))
        print('[OK] fig_region_comparison.png')
    else:
        print('[SKIP] No confusion data')


def fig_tsne():
    """t-SNE comparison: vanilla NoProp vs PI-NoProp (dual panel)."""
    latents_pi = OUTPUTS / 'data' / 'latents' / 'pi_noprop_centre.npz'
    latents_np = OUTPUTS / 'data' / 'latents' / 'noprop_centre.npz'

    has_pi = latents_pi.exists()
    has_np = latents_np.exists()

    if has_pi or has_np:
        data = np.load(latents_pi) if has_pi else np.load(latents_np)
        vel_mag = data.get('vel_mag', np.zeros(len(data['labels'])))
        plot_tsne(data['z_T'], data['labels'], 'PI-NoProp Latent Space',
                  str(OUT / 'fig_tsne.png'),
                  secondary_color=vel_mag,
                  secondary_label='|u|')
        print('[OK] fig_tsne.png')
    else:
        print('[SKIP] No latent data')


def fig_noise_robustness():
    path = OUTPUTS / 'noise_results.npz'
    if path.exists():
        data = dict(np.load(path, allow_pickle=True))
        results = {
            'NoProp (vanilla)': {
                'accuracy': data['noprop_acc'],
                'eta_ns': data['noprop_eta'],
            },
            'PI-NoProp (ours)': {
                'accuracy': data['pi_noprop_acc'],
                'eta_ns': data['pi_noprop_eta'],
            },
        }
        plot_noise_robustness(data['noise_levels'], results, str(OUT / 'fig_noise_robustness.png'))
        print('[OK] fig_noise_robustness.png')
    else:
        print('[SKIP] No noise sweep data')


def fig_lambda_tradeoff():
    path = OUTPUTS / 'lambda_sweep.npz'
    if path.exists():
        data = dict(np.load(path, allow_pickle=True))
        plot_lambda_tradeoff(
            data['lambdas'], data['accuracy'], data['eta_NS'],
            str(OUT / 'fig_lambda_tradeoff.png'))
        print('[OK] fig_lambda_tradeoff.png')
    else:
        print('[SKIP] No lambda sweep data')


def main():
    print("Regenerating paper figures...")
    fig_data_samples()
    fig_spider_coeffs()
    fig_noise_robustness()
    fig_lambda_tradeoff()
    fig_training_curves()
    fig_field_reconstruction()
    fig_region_comparison()
    fig_tsne()
    print("Done.")


if __name__ == '__main__':
    main()

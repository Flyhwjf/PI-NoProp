"""Export and quantify test-set NoProp latents for the 4.6 analysis."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.hit_dataset import create_hit_dataloaders
from src.noprop.model import NoPropModel

SEEDS = (42, 123, 456)
REGIONS = ('low_enstrophy', 'high_enstrophy')
METHODS = (('none', 'No equation'), ('analytic', 'Analytic NS'),
           ('discovered', 'PI-NoProp'))


def run_dir(method, region, seed):
    weight = '0' if method == 'none' else '0p01'
    return ROOT/'outputs/runs'/f'full_ns_v4_{method}_{region}_lambda{weight}_seed{seed}'


@torch.no_grad()
def collect(method, region, seed):
    checkpoint = torch.load(run_dir(method, region, seed)/'checkpoint.pt',
                            map_location='cuda' if torch.cuda.is_available() else 'cpu',
                            weights_only=False)
    config = checkpoint['config']
    config.data.regions = [region]
    config.data.data_dir = 'data/generated_hit_ns'
    config.data.cache_dir = 'data/cache_hit_ns'
    device = torch.device(config.device)
    model = NoPropModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    loader = create_hit_dataloaders(config)[region]['test']
    torch.manual_seed(seed + 10_000)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed + 10_000)
    latents, labels = [], []
    for batch in loader:
        fields = batch['field'].to(device)
        terms = batch.get('ns_terms')
        if terms is not None:
            terms = terms.to(device)
        _, all_latents = model(fields, ns_terms=terms, return_all_latents=True)
        latents.append(all_latents[-1].cpu().numpy())
        labels.append(batch['label'].numpy())
    return np.concatenate(latents), np.concatenate(labels)


def summarize(latents, labels):
    classes = np.unique(labels)
    centroids = np.stack([latents[labels == cls].mean(0) for cls in classes])
    within = np.mean([np.trace(np.cov(latents[labels == cls].T))
                      for cls in classes])
    between = float(np.mean([np.linalg.norm(centroids[i]-centroids[j])
                             for i in range(len(classes))
                             for j in range(i+1, len(classes))]))
    return {'within_class_trace': float(within),
            'between_class_centroid_distance': between,
            'silhouette_128d': float(silhouette_score(latents, labels)),
            'n_samples': int(len(labels))}


def main():
    artifact = {'schema_version': 1, 'protocol': {
        'regions': list(REGIONS), 'methods': [m[0] for m in METHODS],
        'seeds': list(SEEDS), 'split': 'trajectory-disjoint test split',
        'inference_seed': 'model seed + 10000',
        'embedding': 'final z_T, 128 dimensions',
        'projection': 'PCA and t-SNE fitted jointly across methods per region'},
        'results': {}, 'points': {}}
    for region in REGIONS:
        records = {}
        for method, label in METHODS:
            all_z, all_y = [], []
            for seed in SEEDS:
                z, y = collect(method, region, seed)
                all_z.append(z); all_y.append(y)
            z = np.concatenate(all_z); y = np.concatenate(all_y)
            records[method] = (z, y)
            artifact['results'].setdefault(region, {})[method] = summarize(z, y)
        combined = np.concatenate([records[m[0]][0] for m in METHODS])
        pca = PCA(n_components=2, random_state=17).fit_transform(combined)
        tsne = TSNE(n_components=2, perplexity=30, init='pca',
                    learning_rate='auto', random_state=17,
                    max_iter=1000).fit_transform(combined)
        offset = 0
        artifact['points'][region] = {}
        for method, label in METHODS:
            n = len(records[method][0])
            z, y = records[method]
            artifact['points'][region][method] = {
                'label': label, 'labels': y.tolist(),
                'pca': pca[offset:offset+n].tolist(),
                'tsne': tsne[offset:offset+n].tolist()}
            offset += n
    output = ROOT/'outputs/aggregate/full_ns_latent_analysis.json'
    output.write_text(json.dumps(artifact, indent=2), encoding='utf-8')
    print(json.dumps(artifact['results'], indent=2))


if __name__ == '__main__':
    main()

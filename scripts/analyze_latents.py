"""Export and quantify test-set NoProp latents for the 4.6 analysis."""
from __future__ import annotations

import json
import argparse
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
from scripts.experiment_protocol import (ExperimentProtocol, add_protocol_arguments,
                                        load_run, protocol_from_args)

SEEDS = (42, 123, 456)
REGIONS = ('low_enstrophy', 'high_enstrophy')
METHODS = (('none', 'No equation'), ('analytic', 'Analytic NS'),
           ('discovered', 'PI-NoProp'))


def run_dir(method, region, seed, protocol=None):
    protocol = protocol or ExperimentProtocol()
    return ROOT/'outputs/runs'/protocol.run_id(method, region, seed)


@torch.no_grad()
def collect(method, region, seed, protocol=None):
    protocol = protocol or ExperimentProtocol()
    checkpoint, config, _ = load_run(
        run_dir(method, region, seed, protocol), protocol,
        source=method, region=region, seed=seed)
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


def build_parser():
    parser = argparse.ArgumentParser()
    add_protocol_arguments(parser)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    protocol = protocol_from_args(args, parser)
    artifact = {'schema_version': 5, 'protocol': {**protocol.metadata(),
        'model_revision': protocol.model_revision,
        'regions': list(REGIONS), 'methods': [m[0] for m in METHODS],
        'seeds': list(SEEDS), 'split': 'trajectory-disjoint test split',
        'inference_seed': 'model seed + 10000',
        'embedding': 'final z_T, embedding dimension from checkpoint config',
        'projection': 'PCA and t-SNE fitted jointly across methods per region'},
        'results': {}, 'points': {}}
    for region in REGIONS:
        records = {}
        for method, label in METHODS:
            all_z, all_y = [], []
            for seed in SEEDS:
                z, y = collect(method, region, seed, protocol)
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
    output = protocol.aggregate_path(ROOT, 'latent_analysis')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2), encoding='utf-8')
    print(json.dumps(artifact['results'], indent=2))


if __name__ == '__main__':
    main()

"""Audit whether the future-decay target is identifiable from its input."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.hit_dataset import REGION_NAMES, build_learning_cache


def main():
    cache = ROOT/'data/cache_hit_ns'
    build_learning_cache(ROOT/'data/generated_hit_ns', cache)
    metadata = json.loads((cache/'metadata.json').read_text(encoding='utf-8'))
    labels = np.load(cache/'labels.npy')
    splits = np.load(cache/'splits.npy')
    regions = np.load(cache/'regions.npy')
    terms = np.load(cache/'ns_terms.npy')
    stats = np.load(cache/'stats.npz')
    targets = stats['target_values']
    sequences = np.load(cache/'sequences.npy', mmap_mode='r')
    energy = 0.5*np.mean(np.sum(sequences[:, :, :3]**2, axis=2), axis=(2, 3, 4))
    discovered = json.loads(
        (ROOT/'outputs/spider/full_ns_equation.json').read_text(encoding='utf-8'))
    coefficients = np.asarray(discovered['equation']['coefficients'][1:4])
    ns_rate = terms @ coefficients

    result = {'schema_version': 1, 'regions': {}}
    for region, name in enumerate(REGION_NAMES):
        train = (splits == 0) & (regions == region)
        validation = (splits == 1) & (regions == region)
        test = (splits == 2) & (regions == region)
        edges = np.asarray(metadata['class_edges'][name])

        def audit(feature):
            regressor = LinearRegression().fit(feature[train, None], targets[train])
            prediction = regressor.predict(feature[:, None])
            predicted_labels = np.digitize(prediction, edges[1:-1])
            return {
                split: 100*accuracy_score(labels[mask], predicted_labels[mask])
                for split, mask in [('train', train), ('validation', validation),
                                    ('test', test)]}

        first_step_decay = (energy[:, 0]-energy[:, 1])/energy[:, 0]
        result['regions'][name] = {
            'first_frame_discovered_ns_accuracy': audit(ns_rate),
            'two_frame_sanity_accuracy': audit(first_step_decay),
            'ns_rate_target_correlation_test': float(
                np.corrcoef(ns_rate[test], targets[test])[0, 1]),
        }
    output = ROOT/'outputs/aggregate/hit_predictability_audit.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

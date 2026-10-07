"""Test protocol-aware manuscript values using compact synthetic artifacts."""
from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from scripts import sync_paper_results as paper
from scripts.experiment_protocol import ExperimentProtocol


def metric(values=(82.0, 84.0, 86.0)):
    return {'values': list(values), 'mean': float(np.mean(values)),
            'std': float(np.std(values, ddof=1))}


@pytest.fixture
def artifacts():
    protocol = ExperimentProtocol(device='cpu')
    metadata = {**protocol.metadata(), 'seeds': [42, 123, 456]}
    physical = {'accuracy': metric(), 'eta_ns': metric((.1, .2, .3)),
                'eta_div': metric((.02, .04, .06))}
    baseline = {**physical, 'train_seconds': metric((1., 2., 3.)),
                'peak_memory_mb': metric((10., 10., 10.)),
                'parameters': metric((100., 100., 100.))}
    grouped = {
        region: {method: copy.deepcopy(baseline) for method in paper.METHODS}
        for region in paper.REGIONS}
    main = {region: {source: copy.deepcopy(physical)
                     for source in ('none', 'analytic', 'discovered')}
            for region in paper.REGIONS}
    relation = {**physical, 'eta_pp': metric((.1, .2, .3)),
                'eta_energy': metric((.2, .3, .4))}
    # A genuinely separate NS execution is allowed to differ from the main run.
    relation['accuracy'] = metric((81., 83., 85.))
    relation_metadata = {
        **metadata, 'shared_condition_and_decoder_verified': True,
        'run_ids': {variant: [
            protocol.run_id('discovered', 'low_enstrophy', seed,
                            relation=variant, tag='eqablation')
            for seed in (42, 123, 456)] for variant in ('ns', 'ns_pp', 'full')}}
    data = {
        'results': {'protocol': metadata, 'results': main},
        'baselines': {'protocol': metadata, 'results': grouped},
        'lambda_ablation': {'protocol': metadata, 'results': {
            str(weight): copy.deepcopy(physical)
            for weight in (0., .001, .003, .01, .03, .1)}},
        'relation_ablation': {'protocol': relation_metadata, 'results': {
            variant: copy.deepcopy(relation) for variant in ('ns', 'ns_pp', 'full')}},
        'decoder_ablation': {'protocol': metadata, 'results': {
            name: {**copy.deepcopy(physical),
                   'decoder_parameters': metric((100., 100., 100.))}
            for name in ('linear', 'conv')}},
        'noise': {'protocol': metadata, 'results': {
            region: {source: {str(level): copy.deepcopy(physical)
                              for level in (0., .01, .05, .1, .2, .5, 1.)}
                     for source in ('none', 'discovered')}
            for region in paper.REGIONS}},
        'latent_analysis': {'protocol': metadata, 'results': {
            region: {source: {'between_class_centroid_distance': 8.,
                              'silhouette_128d': .1, 'within_class_trace': 80.}
                     for source in ('none', 'analytic', 'discovered')}
            for region in paper.REGIONS}},
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root/'paper').mkdir()
        for name, record in data.items():
            path = protocol.aggregate_path(root, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record), encoding='utf-8')
        with patch.object(paper, 'ROOT', root):
            yield root, protocol, data


def update_artifact(root, protocol, name, data):
    protocol.aggregate_path(root, name).write_text(json.dumps(data), encoding='utf-8')


def test_values_recompute_sample_sd_and_keep_separate_relation_control(artifacts):
    _, protocol, _ = artifacts
    values, sources = paper.collect_values(protocol)
    assert values['lowPiAccuracy'] == '84.00'
    assert values['lowPiAccuracyStd'] == '2.00'
    assert values['lowPiAccuracySummary'] == r'\ensuremath{84.00\pm2.00}'
    assert values['relationNSAccuracy'] == '83.00'
    assert len(sources) == 7
    assert all(len(source['sha256']) == 64 for source in sources.values())


def test_protocol_mismatch_fails_before_generating_values(artifacts):
    root, protocol, data = artifacts
    wrong = copy.deepcopy(data['noise'])
    wrong['protocol']['input_spatial_size'] = 16
    update_artifact(root, protocol, 'noise', wrong)
    with pytest.raises(ValueError, match='incompatible input_spatial_size'):
        paper.collect_values(protocol)


def test_reported_seed_statistics_and_reused_baseline_values_are_checked(artifacts):
    root, protocol, data = artifacts
    wrong = copy.deepcopy(data['baselines'])
    wrong['results']['low_enstrophy']['pi_noprop']['accuracy']['std'] = 1.
    update_artifact(root, protocol, 'baselines', wrong)
    with pytest.raises(ValueError, match='SD differs'):
        paper.collect_values(protocol)
    wrong['results']['low_enstrophy']['pi_noprop']['accuracy'] = metric((81., 83., 85.))
    update_artifact(root, protocol, 'baselines', wrong)
    with pytest.raises(ValueError, match='baseline and main data differ'):
        paper.collect_values(protocol)


def test_relation_provenance_and_verified_components_are_required(artifacts):
    root, protocol, data = artifacts
    wrong = copy.deepcopy(data['relation_ablation'])
    wrong['protocol']['shared_condition_and_decoder_verified'] = False
    update_artifact(root, protocol, 'relation_ablation', wrong)
    with pytest.raises(ValueError, match='matched frozen components'):
        paper.collect_values(protocol)
    wrong['protocol']['shared_condition_and_decoder_verified'] = True
    wrong['protocol']['run_ids']['ns'][0] = 'legacy16_control'
    update_artifact(root, protocol, 'relation_ablation', wrong)
    with pytest.raises(ValueError, match='run provenance'):
        paper.collect_values(protocol)


def test_generation_check_detects_source_changes_even_with_identical_values(artifacts):
    root, protocol, data = artifacts
    with patch('sys.argv', ['sync_paper_results.py', '--device', 'cpu']):
        paper.main()
    with patch('sys.argv', ['sync_paper_results.py', '--device', 'cpu', '--check']):
        paper.main()
        changed = copy.deepcopy(data['noise'])
        changed['protocol']['audit_note'] = 'source metadata changed'
        update_artifact(root, protocol, 'noise', changed)
        with pytest.raises(AssertionError, match='evidence report is stale'):
            paper.main()

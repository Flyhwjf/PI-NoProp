"""Generate manuscript numbers from one complete, size-matched experiment suite.

The generated TeX contains values only. Scientific interpretation remains in
the manuscript, and the source hashes make every displayed value traceable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_protocol import (add_protocol_arguments,
                                        protocol_from_args)

REGIONS = {'low_enstrophy': 'low', 'high_enstrophy': 'high'}
METHODS = {'noprop_reference': 'Ref', 'noprop_ct': 'CT', 'cnn_bp': 'CNN',
           'global_physics_bp': 'Global', 'spider_rate_classifier': 'Rate',
           'noprop_no_equation': 'None', 'noprop_analytic_ns': 'Analytic',
           'pi_noprop': 'Pi'}


def collect_values(protocol):
    artifacts, sources, macros = {}, {}, {}
    for name in ('results', 'baselines', 'noise', 'lambda_ablation',
                 'decoder_ablation', 'relation_ablation', 'latent_analysis'):
        path = protocol.aggregate_path(ROOT, name)
        raw = path.read_bytes()
        data = json.loads(raw)
        metadata = data['protocol']
        for key, expected in (('input_spatial_size', protocol.input_size),
                              ('target_spatial_size', protocol.target_size),
                              ('spatial_context_mode', protocol.context_encoder)):
            if metadata.get(key) != expected:
                raise ValueError(f'{path}: incompatible {key}')
        weight = metadata.get('lambda_weight', metadata.get('lambda_phys'))
        if weight != protocol.lambda_phys:
            raise ValueError(f'{path}: incompatible selected physics weight')
        if metadata.get('seeds') != [42, 123, 456]:
            raise ValueError(f'{path}: expected the three declared seeds')
        artifacts[name] = data
        sources[name] = {'path': str(path.relative_to(ROOT)).replace('\\', '/'),
                         'sha256': hashlib.sha256(raw).hexdigest()}

    def scalar(name, value, digits=2):
        if name in macros or not math.isfinite(float(value)):
            raise ValueError(f'duplicate or non-finite manuscript value: {name}')
        macros[name] = f'{value:.{digits}f}'

    def metric(name, record, digits=2):
        values = record['values']
        if len(values) != 3 or not np.isfinite(values).all():
            raise ValueError(f'{name}: three finite seed values required')
        mean, std = float(np.mean(values)), float(np.std(values, ddof=1))
        if not np.isclose(mean, record['mean'], rtol=1e-8, atol=1e-10):
            raise ValueError(f'{name}: reported mean differs from seed values')
        if record.get('std') is not None and not np.isclose(
                std, record['std'], rtol=1e-8, atol=1e-10):
            raise ValueError(f'{name}: reported SD differs from seed values')
        scalar(name, mean, digits)
        scalar(name + 'Std', std, digits)
        macros[name + 'Summary'] = (
            r'\ensuremath{' + f'{mean:.{digits}f}' + r'\pm'
            + f'{std:.{digits}f}' + '}')

    scalar('selectedPhysicsWeight', protocol.lambda_phys, 1)
    scalar('modelInputSize', protocol.input_size, 0)
    scalar('modelTargetSize', protocol.target_size, 0)
    main = artifacts['results']['results']
    baselines = artifacts['baselines']['results']
    for region, short in REGIONS.items():
        for method, label in METHODS.items():
            record = baselines[region][method]
            metric(short + label + 'Accuracy', record['accuracy'])
            for field, suffix, digits in (
                    ('eta_ns', 'NS', 3), ('eta_div', 'Div', 3),
                    ('train_seconds', 'Seconds', 2),
                    ('peak_memory_mb', 'Memory', 1), ('parameters', 'Parameters', 0)):
                if record[field]['mean'] is not None:
                    metric(short + label + suffix, record[field], digits)
            source = {'None': 'none', 'Analytic': 'analytic',
                      'Pi': 'discovered'}.get(label)
            if source:
                for key in ('accuracy', 'eta_ns', 'eta_div'):
                    if record[key]['values'] != main[region][source][key]['values']:
                        raise ValueError(f'{region}/{label}: baseline and main data differ')
        pi, none = main[region]['discovered'], main[region]['none']
        scalar(short + 'AccuracyGain', pi['accuracy']['mean']-none['accuracy']['mean'])
        scalar(short + 'NSReduction',
               100*(1-pi['eta_ns']['mean']/none['eta_ns']['mean']), 1)
        scalar(short + 'RateGap', baselines[region]['spider_rate_classifier']
               ['accuracy']['mean']-pi['accuracy']['mean'])
        scalar(short + 'AnalyticGap', pi['accuracy']['mean']
               -main[region]['analytic']['accuracy']['mean'])

    for value, label in ((0.0, 'Zero'), (0.001, 'Tiny'), (0.003, 'Small'),
                         (0.01, 'Standard'), (0.03, 'Medium'), (0.1, 'Selected')):
        record = artifacts['lambda_ablation']['results'][str(value)]
        for field, suffix, digits in (('accuracy', 'Accuracy', 2),
                                      ('eta_ns', 'NS', 3), ('eta_div', 'Div', 3)):
            metric('lambda' + label + suffix, record[field], digits)
    accuracies = [record['accuracy']['mean'] for record in
                  artifacts['lambda_ablation']['results'].values()]
    scalar('lambdaAccuracyMin', min(accuracies))
    scalar('lambdaAccuracyMax', max(accuracies))

    for variant, label in (('ns', 'NS'), ('ns_pp', 'PP'), ('full', 'Full')):
        record = artifacts['relation_ablation']['results'][variant]
        for field, suffix, digits in (('accuracy', 'Accuracy', 2), ('eta_ns', 'NS', 3),
                                      ('eta_div', 'Div', 3), ('eta_pp', 'PP', 3),
                                      ('eta_energy', 'Energy', 3)):
            metric('relation' + label + suffix, record[field], digits)
    # The relation study retrains its NS control; seeded AMP/cuDNN execution
    # need not reproduce an earlier training run bit-for-bit. Validate the
    # declared run provenance and matched frozen components, not equality of
    # results from two distinct executions. The baseline/main equality above
    # still applies because those tables reuse the same saved runs.
    relations = artifacts['relation_ablation']['protocol']
    expected_runs = {
        variant: [protocol.run_id('discovered', 'low_enstrophy', seed,
                                 relation=variant, tag='eqablation')
                  for seed in (42, 123, 456)]
        for variant in ('ns', 'ns_pp', 'full')}
    if relations.get('run_ids') != expected_runs:
        raise ValueError('relation ablation run provenance is missing or incompatible')
    if not relations.get('shared_condition_and_decoder_verified'):
        raise ValueError('relation variants require verified matched frozen components')

    for architecture, label in (('linear', 'Linear'), ('conv', 'Conv')):
        record = artifacts['decoder_ablation']['results'][architecture]
        for field, suffix, digits in (('accuracy', 'Accuracy', 2),
                                      ('eta_ns', 'NS', 3), ('eta_div', 'Div', 3),
                                      ('decoder_parameters', 'Parameters', 0)):
            metric('decoder' + label + suffix, record[field], digits)
        scalar('decoder' + label + 'Millions', record['decoder_parameters']['mean']/1e6)
    decoders = artifacts['decoder_ablation']['results']
    scalar('decoderParameterRatio', decoders['linear']['decoder_parameters']['mean']
           /decoders['conv']['decoder_parameters']['mean'], 1)

    levels = ((0.0, 'Clean'), (0.01, 'One'), (0.05, 'Five'), (0.1, 'Ten'),
              (0.2, 'Twenty'), (0.5, 'Fifty'), (1.0, 'Hundred'))
    noise = artifacts['noise']['results']
    latent = artifacts['latent_analysis']['results']
    for region, short in REGIONS.items():
        for method, label in (('none', 'None'), ('discovered', 'Pi')):
            for value, level in levels:
                record = noise[region][method][str(value)]
                metric(short + label + 'Noise' + level + 'Accuracy', record['accuracy'])
                metric(short + label + 'Noise' + level + 'NS', record['eta_ns'], 3)
        clean = noise[region]['discovered']['0.0']['accuracy']['mean']
        five = noise[region]['discovered']['0.05']['accuracy']['mean']
        scalar(short + 'NoiseFiveDrop', clean-five)
        for method, label in (('none', 'None'), ('analytic', 'Analytic'),
                              ('discovered', 'Pi')):
            record = latent[region][method]
            scalar(short + label + 'Centroid', record['between_class_centroid_distance'], 3)
            scalar(short + label + 'Silhouette', record['silhouette_128d'], 3)
            scalar(short + label + 'Spread', record['within_class_trace'])
        scalar(short + 'SpreadChange', 100*(latent[region]['discovered']['within_class_trace']
               /latent[region]['none']['within_class_trace']-1), 1)
    return macros, sources


def main():
    parser = argparse.ArgumentParser()
    add_protocol_arguments(parser)
    parser.add_argument('--check', action='store_true',
                        help='verify generated TeX and source hashes without writing')
    args = parser.parse_args()
    protocol = protocol_from_args(args, parser)
    macros, sources = collect_values(protocol)
    text = ('% Generated by scripts/sync_paper_results.py; values are mean and sample SD.\n'
            + f'% Protocol: {protocol.prefix}; input {protocol.input_size}, '
              f'target {protocol.target_size}, lambda {protocol.lambda_phys:g}.\n'
            + ''.join('\\newcommand{\\' + name + '}{' + value + '}\n'
                      for name, value in sorted(macros.items())))
    destination = ROOT/'paper/experiment_values.tex'
    report_path = protocol.aggregate_path(ROOT, 'paper_evidence')
    report = {'protocol': protocol.metadata(), 'sources': sources,
              'values': macros, 'generated_tex_sha256': hashlib.sha256(
                  text.encode('utf-8')).hexdigest()}
    if args.check:
        if destination.read_text(encoding='utf-8') != text:
            raise AssertionError('manuscript numerical macros are stale')
        if json.loads(report_path.read_text(encoding='utf-8')) != report:
            raise AssertionError('paper evidence report is stale')
    else:
        destination.write_text(text, encoding='utf-8')
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'{"Verified" if args.check else "Generated"} {len(macros)} '
          f'manuscript values from {len(sources)} compatible artifacts: {destination}')


if __name__ == '__main__':
    main()

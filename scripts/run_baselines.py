"""Trajectory-disjoint baselines corresponding to the original manuscript table."""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines.cnn import SimpleCNN
from src.config import PINoPropConfig
from src.data.hit_dataset import create_hit_dataloaders
from src.baselines.noprop_reference import ReferenceNoProp3D, ContinuousNoProp3D
from src.noprop.model import NoPropModel
from src.physics.temporal_ns_loss import TemporalNSPhysicsLoss
from src.training.local_trainer import configure_torch
from scripts.experiment_protocol import (ExperimentProtocol, add_protocol_arguments,
                                        load_run, make_decoder, protocol_from_args,
                                        protect_legacy_output, validate_metadata)

REGIONS = ('low_enstrophy', 'high_enstrophy')
SEEDS = (42, 123, 456)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_config(region, coefficients=None, protocol=None):
    protocol = protocol or ExperimentProtocol()
    config = PINoPropConfig()
    protocol.apply_config(config)
    config.data.regions = [region]
    config.physics.n_time = 9
    config.physics.beta = 4.0
    config.physics.n_test_functions = 8
    config.physics.condition_coefficients = coefficients
    config.physics.use_full_ns = True
    config.physics.use_continuity = True
    config.physics.use_pressure_poisson = False
    config.physics.use_energy = False
    return config


@torch.no_grad()
def evaluate_classifier(model, loader, device):
    model.eval(); correct = total = 0
    for batch in loader:
        logits = model(batch['field'].to(device, non_blocking=True))
        labels = batch['label'].to(device, non_blocking=True)
        correct += int((logits.argmax(-1) == labels).sum()); total += len(labels)
    return 100*correct/max(total, 1)


def train_cnn(region, seed, loaders, config, epochs=40):
    seed_all(seed); device = torch.device(config.device)
    model = SimpleCNN(config.data.n_channels, config.data.n_classes,
                      config.data.subdomain_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_accuracy, best_state = -1.0, None
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for batch in loaders['train']:
            fields = batch['field'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(fields), labels)
            loss.backward(); optimizer.step()
        accuracy = evaluate_classifier(model, loaders['val'], device)
        if accuracy > best_accuracy:
            best_accuracy = accuracy; best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return {
        'method': 'CNN (BP)', 'region': region, 'seed': seed,
        'accuracy': evaluate_classifier(model, loaders['test'], device),
        'eta_ns': None, 'eta_div': None,
        'train_seconds': time.perf_counter()-started,
        'peak_memory_mb': (torch.cuda.max_memory_allocated()/1024**2
                           if device.type == 'cuda' else 0.0),
        'parameters': sum(p.numel() for p in model.parameters()),
    }


@torch.no_grad()
def evaluate_reference_noprop(model, loader, device, seed):
    devices = ([torch.cuda.current_device()] if device.type == 'cuda' else [])
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed+10_000)
        if device.type == 'cuda': torch.cuda.manual_seed_all(seed+10_000)
        model.eval(); correct = total = 0
        for batch in loader:
            fields = batch['field'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            correct += int((model(fields).argmax(-1) == labels).sum())
            total += len(labels)
    return 100*correct/max(total, 1)


def train_reference_noprop(region, seed, loaders, config, steps_per_block=100):
    seed_all(seed); device = torch.device(config.device)
    model = ReferenceNoProp3D(
        n_classes=config.data.n_classes, n_blocks=config.diffusion.T,
        eta=config.diffusion.eta).to(device)
    optimizers = [torch.optim.AdamW(
        step.parameters(), lr=1e-3, weight_decay=1e-4)
        for step in model.steps]
    updates = torch.zeros(model.n_blocks, dtype=torch.long)
    iterator = iter(loaders['train'])
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter(); model.train()
    while int(updates.min()) < steps_per_block:
        eligible = torch.where(updates < steps_per_block)[0]
        block_index = int(eligible[torch.randint(len(eligible), ())])
        try: batch = next(iterator)
        except StopIteration:
            iterator = iter(loaders['train']); batch = next(iterator)
        fields = batch['field'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        optimizer = optimizers[block_index]
        optimizer.zero_grad(set_to_none=True)
        loss = model.local_loss(fields, labels, block_index)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.steps[block_index].parameters(), 1.0)
        optimizer.step(); updates[block_index] += 1
    return {
        'method': 'Reference NoProp (3-D)', 'region': region, 'seed': seed,
        'accuracy': evaluate_reference_noprop(model, loaders['test'], device, seed),
        'eta_ns': None, 'eta_div': None,
        'train_seconds': time.perf_counter()-started,
        'peak_memory_mb': (torch.cuda.max_memory_allocated()/1024**2
                           if device.type == 'cuda' else 0.0),
        'parameters': sum(parameter.numel() for parameter in model.parameters()),
        'block_updates': updates.tolist(),
        'protocol': 'one-hot targets; independent 3-D encoder and optimizer per block',
    }


@torch.no_grad()
def evaluate_noprop_ct(model, loader, device, seed):
    devices = ([torch.cuda.current_device()] if device.type == 'cuda' else [])
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed+10_000)
        if device.type == 'cuda': torch.cuda.manual_seed_all(seed+10_000)
        model.eval(); correct = total = 0
        for batch in loader:
            fields = batch['field'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            latent = model.integrate(fields, steps=10, adjoint=False)
            correct += int((latent.argmax(-1) == labels).sum()); total += len(labels)
    return 100*correct/max(total, 1)


def train_noprop_ct(region, seed, loaders, config, updates=1000):
    seed_all(seed); device = torch.device(config.device)
    model = ContinuousNoProp3D(n_classes=config.data.n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    iterator = iter(loaders['train'])
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter(); model.train()
    for _ in range(updates):
        try: batch = next(iterator)
        except StopIteration:
            iterator = iter(loaders['train']); batch = next(iterator)
        fields = batch['field'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = model.flow_matching_loss(fields, labels)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return {
        'method': 'NoProp-CT (3-D conditional flow)', 'region': region, 'seed': seed,
        'accuracy': evaluate_noprop_ct(model, loaders['test'], device, seed),
        'eta_ns': None, 'eta_div': None,
        'train_seconds': time.perf_counter()-started,
        'peak_memory_mb': (torch.cuda.max_memory_allocated()/1024**2
                           if device.type == 'cuda' else 0.0),
        'parameters': sum(parameter.numel() for parameter in model.parameters()),
        'updates': updates, 'ode_solver': 'fixed-step RK4, 10 steps',
        'protocol': 'continuous-time conditional flow matching with one-hot endpoints',
    }


class GlobalPhysicsClassifier(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        backbone = NoPropModel(config)
        self.encoder = backbone.encoder
        self.physics_encoder = backbone.physics_encoder
        self.condition_fusion = backbone.condition_fusion
        self.register_buffer('physics_coefficients', backbone.physics_coefficients)
        self.register_buffer('physics_mean', backbone.physics_mean)
        self.register_buffer('physics_std', backbone.physics_std)
        self.head = torch.nn.Linear(config.noprop.condition_dim, config.data.n_classes)

    def encode(self, fields, terms):
        spatial = self.encoder(fields)
        rate = ((terms@self.physics_coefficients-self.physics_mean)
                / self.physics_std).unsqueeze(-1)
        physical = self.physics_encoder(rate)
        return self.condition_fusion(torch.cat([spatial, physical], dim=-1))

    def forward(self, fields, terms):
        return self.head(self.encode(fields, terms))


@torch.no_grad()
def evaluate_global(model, decoder, physics, loader, device, include_physics=False):
    model.eval(); decoder.eval(); correct = total = 0; ns = div = batches = 0
    for batch in loader:
        fields = batch['field'].to(device); terms = batch['ns_terms'].to(device)
        labels = batch['label'].to(device)
        condition = model.encode(fields, terms)
        correct += int((model.head(condition).argmax(-1) == labels).sum())
        total += len(labels)
        if include_physics:
            metrics = physics.evaluate_metrics(decoder(condition))
            ns += metrics['eta_ns']; div += metrics['eta_div']; batches += 1
    result = {'accuracy': 100*correct/max(total, 1)}
    if include_physics:
        result.update(eta_ns=ns/max(batches, 1), eta_div=div/max(batches, 1))
    return result


def train_global_physics(region, seed, loaders, config, artifact, epochs=30):
    seed_all(seed); device = torch.device(config.device)
    model = GlobalPhysicsClassifier(config).to(device)
    decoder = make_decoder(config).to(device)
    physics = TemporalNSPhysicsLoss(config, artifact).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters())+list(decoder.parameters()), lr=1e-3,
        weight_decay=1e-4)
    best_accuracy, best_state, stale = -1.0, None, 0
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(epochs):
        model.train(); decoder.train()
        for batch in loaders['train']:
            fields = batch['field'].to(device); terms = batch['ns_terms'].to(device)
            labels = batch['label'].to(device); sequence = batch['sequence'].to(device)
            optimizer.zero_grad(set_to_none=True)
            condition = model.encode(fields, terms)
            decoded = decoder(condition)
            physical_loss, _ = physics(decoded)
            loss = (F.cross_entropy(model.head(condition), labels)
                    + 0.05*F.mse_loss(decoded, sequence)
                    + config.physics.lambda_weight*physical_loss)
            loss.backward(); torch.nn.utils.clip_grad_norm_(
                list(model.parameters())+list(decoder.parameters()), 1.0)
            optimizer.step()
        accuracy = evaluate_global(model, decoder, physics, loaders['val'], device)['accuracy']
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = (copy.deepcopy(model.state_dict()),
                          copy.deepcopy(decoder.state_dict()))
            stale = 0
        else:
            stale += 1
        if stale >= 8: break
    model.load_state_dict(best_state[0]); decoder.load_state_dict(best_state[1])
    test = evaluate_global(model, decoder, physics, loaders['test'], device, True)
    return {
        'method': 'Global physics-informed BP', 'region': region, 'seed': seed,
        **test, 'train_seconds': time.perf_counter()-started,
        'peak_memory_mb': (torch.cuda.max_memory_allocated()/1024**2
                           if device.type == 'cuda' else 0.0),
        'parameters': (sum(p.numel() for p in model.parameters())
                       + sum(p.numel() for p in decoder.parameters())),
    }


def spider_rate_classifier(region, seed, loaders, coefficients):
    started = time.perf_counter()
    train, test = loaders['train'].dataset, loaders['test'].dataset
    c = np.asarray(coefficients, dtype=np.float64)
    x_train = train.ns_terms.numpy().astype(np.float64)@c
    x_test = test.ns_terms.numpy().astype(np.float64)@c
    classifier = make_pipeline(
        StandardScaler(), LogisticRegression(C=100.0, max_iter=5000,
                                              random_state=seed))
    classifier.fit(x_train[:, None], train.labels.numpy())
    prediction = classifier.predict(x_test[:, None])
    return {
        'method': 'SPIDER rate + classifier', 'region': region, 'seed': seed,
        'accuracy': 100*accuracy_score(test.labels.numpy(), prediction),
        'eta_ns': None, 'eta_div': None,
        'train_seconds': time.perf_counter()-started,
        'peak_memory_mb': 0.0, 'parameters': 10,
    }


def summarize(records, key):
    values = [record[key] for record in records if record.get(key) is not None]
    if not values: return {'values': [], 'mean': None, 'std': None}
    return {'values': values, 'mean': float(np.mean(values)),
            'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}


def build_parser():
    parser = argparse.ArgumentParser()
    add_protocol_arguments(parser)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--baselines-only', action='store_true',
                        help='train/aggregate standalone baselines without requiring main NoProp runs')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    protocol = protocol_from_args(args, parser)
    artifact = json.loads((ROOT/'outputs/spider/full_ns_equation.json').read_text())
    coefficients = artifact['equation']['coefficients'][1:4]
    all_records = {region: {} for region in REGIONS}
    for region in REGIONS:
        config = make_config(region, coefficients, protocol)
        configure_torch(config)
        loaders = create_hit_dataloaders(config)[region]
        for seed in SEEDS:
            config.seed = seed
            for name, function in (
                ('noprop_reference', lambda: train_reference_noprop(
                    region, seed, loaders, config)),
                ('noprop_ct', lambda: train_noprop_ct(
                    region, seed, loaders, config)),
                ('cnn_bp', lambda: train_cnn(region, seed, loaders, config)),
                ('global_physics_bp', lambda: train_global_physics(
                    region, seed, loaders, config, artifact)),
                ('spider_rate_classifier', lambda: spider_rate_classifier(
                    region, seed, loaders, coefficients))):
                path = (ROOT/'outputs/runs'/
                        protocol.baseline_run_id(name, region, seed)/'metrics.json')
                if path.exists() and not args.overwrite:
                    record = json.loads(path.read_text())
                    validate_metadata(record.get('experiment_protocol', {}), protocol,
                                      legacy_ok=True)
                    if record.get('region') != region or record.get('seed') != seed:
                        raise ValueError(f'baseline identity mismatch: {path}')
                else:
                    protect_legacy_output(path, protocol)
                    print('run', name, region, seed, flush=True)
                    record = function(); path.parent.mkdir(parents=True, exist_ok=True)
                    record['experiment_protocol'] = protocol.metadata()
                    path.write_text(json.dumps(record, indent=2))
                    (path.parent/'config.json').write_text(
                        json.dumps(asdict(config), indent=2), encoding='utf-8')
                all_records[region].setdefault(name, []).append(record)

        main_methods = (() if args.baselines_only else
                        (('noprop_no_equation', 'none'),
                             ('noprop_analytic_ns', 'analytic'),
                             ('pi_noprop', 'discovered')))
        for name, source in main_methods:
            records = []
            for seed in SEEDS:
                path = (ROOT/'outputs/runs'/
                        protocol.run_id(source, region, seed)/
                        'metrics.json')
                _, checkpoint_config, raw = load_run(
                    path.parent, protocol, source=source, region=region, seed=seed)
                counted_model = NoPropModel(checkpoint_config)
                counted_decoder = make_decoder(checkpoint_config)
                records.append({
                    'method': name, 'region': region, 'seed': seed,
                    'accuracy': raw['test']['accuracy'],
                    'eta_ns': raw['test']['eta_ns'], 'eta_div': raw['test']['eta_div'],
                    'train_seconds': raw['block_train_seconds'],
                    'peak_memory_mb': raw['peak_memory_mb'],
                    'parameters': (sum(p.numel() for p in counted_model.parameters())
                                   + sum(p.numel() for p in counted_decoder.parameters())),
                })
            all_records[region][name] = records

    aggregate = {'schema_version': 5, 'protocol': {**protocol.metadata(), 'seeds': list(SEEDS),
                 'trajectory_disjoint': True,
                 'model_revision': protocol.model_revision,
                 'readout': 'cosine similarity to frozen label embeddings',
                 'reference_noprop': 'one-hot, independent 3-D encoder per local block',
                 'noprop_ct': 'continuous conditional flow matching; RK4 inference',
                 'label_and_physics_target': 'centred target cube',
                 'global_bp_lambda_weight': protocol.lambda_phys},
                 'results': {}}
    for region, methods in all_records.items():
        aggregate['results'][region] = {}
        for name, records in methods.items():
            aggregate['results'][region][name] = {
                key: summarize(records, key) for key in
                ('accuracy', 'eta_ns', 'eta_div', 'train_seconds',
                 'peak_memory_mb', 'parameters')}
    output = protocol.aggregate_path(ROOT, 'baselines')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(aggregate, indent=2))
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == '__main__':
    main()

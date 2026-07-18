"""Trajectory-disjoint baselines corresponding to the original manuscript table."""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
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
from src.decoder import TemporalPhysicsDecoder
from src.noprop.model import NoPropModel
from src.physics.temporal_ns_loss import TemporalNSPhysicsLoss
from src.training.local_trainer import configure_torch

REGIONS = ('low_enstrophy', 'high_enstrophy')
SEEDS = (42, 123, 456)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_config(region, coefficients=None):
    config = PINoPropConfig()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.data.data_dir = 'data/generated_hit_ns'
    config.data.cache_dir = 'data/cache_hit_ns'
    config.data.regions = [region]
    config.data.subdomain_size = 16
    config.data.n_subdomains = 64
    config.data.n_classes = 5
    config.data.batch_size = 32
    config.noprop.normalize_condition = True
    config.physics.n_time = 9
    config.physics.beta = 4.0
    config.physics.n_test_functions = 8
    config.physics.physics_grid_size = 16
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
    model = SimpleCNN(4, 5, 16).to(device)
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
    decoder = TemporalPhysicsDecoder(128, 32, 9, 4).to(device)
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
                    + 0.01*physical_loss)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    artifact = json.loads((ROOT/'outputs/spider/full_ns_equation.json').read_text())
    coefficients = artifact['equation']['coefficients'][1:4]
    all_records = {region: {} for region in REGIONS}
    for region in REGIONS:
        config = make_config(region, coefficients)
        configure_torch(config)
        loaders = create_hit_dataloaders(config)[region]
        for seed in SEEDS:
            for name, function in (
                ('cnn_bp', lambda: train_cnn(region, seed, loaders, config)),
                ('global_physics_bp', lambda: train_global_physics(
                    region, seed, loaders, config, artifact)),
                ('spider_rate_classifier', lambda: spider_rate_classifier(
                    region, seed, loaders, coefficients))):
                path = ROOT/'outputs/runs'/f'full_ns_v4_{name}_{region}_seed{seed}'/'metrics.json'
                if path.exists() and not args.overwrite:
                    record = json.loads(path.read_text())
                else:
                    print('run', name, region, seed, flush=True)
                    record = function(); path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(record, indent=2))
                all_records[region].setdefault(name, []).append(record)

        for name, source in (('noprop_vanilla', 'none'), ('pi_noprop', 'discovered')):
            records = []
            weight = '0' if source == 'none' else '0p01'
            for seed in SEEDS:
                path = (ROOT/'outputs/runs'/
                        f'full_ns_v4_{source}_{region}_lambda{weight}_seed{seed}'/
                        'metrics.json')
                raw = json.loads(path.read_text())
                records.append({
                    'method': name, 'region': region, 'seed': seed,
                    'accuracy': raw['test']['accuracy'],
                    'eta_ns': raw['test']['eta_ns'], 'eta_div': raw['test']['eta_div'],
                    'train_seconds': raw['block_train_seconds'],
                    'peak_memory_mb': raw['peak_memory_mb'],
                })
            all_records[region][name] = records

    aggregate = {'schema_version': 2, 'protocol': {'seeds': list(SEEDS),
                 'trajectory_disjoint': True,
                 'model_revision': 'trainable-physics-condition-fusion',
                 'unaffected_baselines': [
                     'cnn_bp', 'global_physics_bp', 'spider_rate_classifier']},
                 'results': {}}
    for region, methods in all_records.items():
        aggregate['results'][region] = {}
        for name, records in methods.items():
            aggregate['results'][region][name] = {
                key: summarize(records, key) for key in
                ('accuracy', 'eta_ns', 'eta_div', 'train_seconds', 'peak_memory_mb')}
    output = ROOT/'outputs/aggregate/full_ns_baselines.json'
    output.write_text(json.dumps(aggregate, indent=2))
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == '__main__':
    main()

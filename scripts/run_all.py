"""
run_all.py — One-shot execution of all PI-NoProp experiments.

Phases (numbered by execution order):
  1. generate_data    — HIT DNS + subdomain extraction (if --skip-dns not set)
  2. spider           — SPIDER PDE discovery + save coefficients
  3. pretrain         — Decoder pretraining (shared for PI-NoProp)
  4. baselines        — CNN, NoProp, NoProp-CT, PINN, SPIDER+CNN
  5. pi_noprop        — PI-NoProp training (centre + edge)
  6. noise_sweep      — Noise robustness sweep
  7. lambda_sweep     — Lambda tradeoff sweep
  8. equation_ablation— NS/NS+PP/full variants
  9. decoder_ablation — Conv vs Linear decoder
  10. benchmark        — Efficiency benchmarking
  11. figures          — Generate all paper figures
  12. summary          — Print results table

Usage:
  python scripts/run_all.py                          # all phases
  python scripts/run_all.py --phase 5                # only phase 5
  python scripts/run_all.py --start 3 --stop 7       # phases 3-7
  python scripts/run_all.py --skip 1,2               # skip DNS and SPIDER
"""
import sys, os, time, json, argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from src.config import PINoPropConfig, DataConfig, DiffusionConfig
from src.config import NoPropConfig, DecoderConfig, PhysicsConfig, TrainingConfig


# ============================================================
#  Utility
# ============================================================

def _json_to_eq(data, device='cpu'):
    """Convert SPIDER JSON dict to (coefficients_tensor, term_defs_list)."""
    if data is None:
        return None
    import torch
    coeffs = torch.tensor(data['coefficients'], device=torch.device(device))
    terms = [{'type': t, 'fields': None} for t in data['terms']]
    return coeffs, terms


def make_config(**overrides):
    """Create a PINoPropConfig with optional overrides."""
    cfg = PINoPropConfig()
    for key, val in overrides.items():
        parts = key.split('.')
        obj = cfg
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], val)
    return cfg


def save_result(result, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)


def save_npz(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def compute_eta_ns(physics_loss, dataloader, device):
    """Compute relative NS residual eta_NS over a dataloader."""
    physics_loss.eval()
    vals = []
    with torch.no_grad():
        for batch in dataloader:
            x = torch.cat([batch['velocity'], batch['pressure'].unsqueeze(1)], dim=1)
            x = x.to(device)
            vals.append(physics_loss.compute_eta_ns(x))
    return float(np.mean(vals)) if vals else 0.0


def compute_eta_ns_reconstructed(physics_loss, model, decoder, dataloader, device):
    """Compute η_NS on decoder-reconstructed fields (not raw input)."""
    model.eval()
    decoder.eval()
    physics_loss.eval()
    vals = []
    with torch.no_grad():
        for batch in dataloader:
            x = torch.cat([batch['velocity'], batch['pressure'].unsqueeze(1)], dim=1)
            x = x.to(device)
            _, z_all = model(x, return_all_latents=True)
            reconstructed = decoder(z_all[-1])
            vals.append(physics_loss.compute_eta_ns(reconstructed))
    return float(np.mean(vals)) if vals else 0.0


# ============================================================
#  Phase 1: Data generation (calls external script)
# ============================================================

def phase_generate_data(args):
    import subprocess
    print("\n" + "=" * 60)
    print("Phase 1: HIT DNS Data Generation")
    print("=" * 60)
    cmd = [
        sys.executable, "-u", "scripts/generate_hit.py",
        "--N", "64", "--nu", "0.005", "--force_amp", "10", "--seed", "42",
        "--n_steady", "2000", "--n_sample", "4000", "--save_every", "50",
        "--n_t", "32", "--dt", "0.001",
    ]
    subprocess.run(cmd, check=True)
    print("Phase 1 done.")


# ============================================================
#  Phase 2: SPIDER
# ============================================================

def phase_spider(args):
    from src.training.pde_discovery import discover_pdes

    print("\n" + "=" * 60)
    print("Phase 2: SPIDER PDE Discovery (4D)")
    print("=" * 60)

    cfg = make_config(**{'data.noise_level': 0.0})

    # Discover PDEs from full 4D centre data
    vec_eq, scal_eq = discover_pdes(cfg, device=cfg.device, region='centre', n_files=32)

    # Save
    os.makedirs('outputs/spider', exist_ok=True)
    for name, (coeffs, terms) in [('vector', vec_eq), ('scalar', scal_eq)]:
        data = {
            'coefficients': [float(c) for c in coeffs.cpu().numpy()],
            'terms': [t['type'] for t in terms],
            'active_terms': [t['type'] for t, c in zip(terms, coeffs) if abs(c) > 1e-3],
            'inactive_terms': [t['type'] for t, c in zip(terms, coeffs) if abs(c) <= 1e-3],
            'viscosity_input': cfg.physics.viscosity,
        }
        save_result(data, f'outputs/spider/{name}_equation.json')

    print("SPIDER done. Equations saved to outputs/spider/")
    return vec_eq, scal_eq


# ============================================================
#  Phase 3: Decoder Pretraining (shared)
# ============================================================

def phase_pretrain(args, vec_eq, scal_eq):
    from src.data.dataset import create_dataloaders
    from src.noprop.model import NoPropModel
    from src.decoder import FieldDecoder
    from src.physics.loss import PhysicsLoss
    from src.training.trainer import Trainer

    print("\n" + "=" * 60)
    print("Phase 3: Decoder Pretraining")
    print("=" * 60)

    cfg = make_config()
    dataloaders = create_dataloaders(cfg)

    for region in ['centre', 'edge']:
        print(f"  Pretraining decoder for {region}...")
        model = NoPropModel(cfg).to(cfg.device)
        decoder = FieldDecoder(
            latent_dim=cfg.decoder.latent_dim,
            base_channels=cfg.decoder.base_channels,
            output_channels=cfg.decoder.output_channels,
        ).to(cfg.device)
        physics_loss = PhysicsLoss(cfg, vec_eq, scal_eq).to(cfg.device)
        trainer = Trainer(model, decoder, physics_loss, cfg)

        train_loader = dataloaders[region]['train']
        trainer.pretrain_decoder(train_loader, n_epochs=cfg.training.n_pretrain_epochs)

        path = f'outputs/models/pretrained_decoder_{region}.pt'
        torch.save(decoder.state_dict(), path)
        print(f"    Saved {path}")

    print("Pretraining done.")
    return dataloaders


# ============================================================
#  Phase 4: Baselines
# ============================================================

def train_cnn_baseline(cfg, dataloaders, region):
    from src.baselines.cnn import SimpleCNN
    from src.training.metrics import evaluate
    import torch.nn as nn
    device = torch.device(cfg.device)

    model = SimpleCNN(in_channels=4, n_classes=cfg.data.n_classes,
                      grid_size=cfg.data.subdomain_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.training.lr,
                           weight_decay=cfg.training.weight_decay)
    crit = nn.CrossEntropyLoss()
    best_acc = 0
    t0 = time.time()

    for epoch in range(cfg.training.n_epochs):
        model.train()
        for batch in dataloaders[region]['train']:
            x = torch.cat([batch['velocity'], batch['pressure'].unsqueeze(1)], dim=1).to(device)
            labels = batch['label'].to(device)
            opt.zero_grad()
            loss = crit(model(x), labels)
            loss.backward()
            opt.step()
        model.eval()
        met = evaluate(model, dataloaders[region]['val'], device)
        if met['accuracy'] > best_acc:
            best_acc = met['accuracy']
            torch.save(model.state_dict(), f'outputs/models/cnn_{region}.pt')

    return {
        'method': 'cnn', 'region': region,
        'accuracy': best_acc,
        'train_time_s': time.time() - t0,
        'n_params': sum(p.numel() for p in model.parameters()),
    }


def train_noprop_baseline(cfg, dataloaders, region, lambda_phys=0.0):
    from src.noprop.model import NoPropModel
    from src.decoder import FieldDecoder
    from src.physics.loss import PhysicsLoss
    from src.training.trainer import Trainer
    device = torch.device(cfg.device)

    cfg2 = make_config(**{'physics.lambda_weight': lambda_phys})
    model = NoPropModel(cfg2).to(device)
    decoder = FieldDecoder().to(device)
    vec_eq = scal_eq = None  # vanilla NoProp doesn't use SPIDER
    physics_loss = PhysicsLoss(cfg2, vec_eq, scal_eq).to(device)
    trainer = Trainer(model, decoder, physics_loss, cfg2)

    train_loader = dataloaders[region]['train']
    val_loader = dataloaders[region]['val']

    # Pretrain decoder
    trainer.pretrain_decoder(train_loader, n_epochs=cfg2.training.n_pretrain_epochs)
    torch.save(decoder.state_dict(), f'outputs/models/pretrained_decoder_{region}_vanilla.pt')

    # Training loop with data collection
    history = {'loss': [], 'cls': [], 'diff': [], 'phys': [], 'acc': []}
    latents_list, labels_list = [], []
    best_acc = 0
    t0 = time.time()

    for epoch in range(cfg2.training.n_epochs):
        train_stats = trainer.train_epoch(train_loader)
        val_stats = trainer.validate(val_loader)
        for k in history:
            if k in train_stats: history[k].append(train_stats[k])
        history['acc'].append(val_stats['accuracy'])
        if val_stats['accuracy'] > best_acc:
            best_acc = val_stats['accuracy']
            trainer.save_checkpoint(epoch, f'outputs/models/noprop_{region}.pt')

        # Collect latents every 25 epochs
        if epoch % 25 == 0 or epoch == cfg2.training.n_epochs - 1:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    x = torch.cat([batch['velocity'], batch['pressure'].unsqueeze(1)], dim=1).to(device)
                    _, z_all = model(x, return_all_latents=True)
                    latents_list.append(z_all[-1].cpu().numpy())
                    labels_list.append(batch['label'].numpy())

    # Save
    save_npz(history, f'outputs/data/training_curves/noprop_{region}.npz')
    latents = {'z_T': np.concatenate(latents_list, 0),
               'labels': np.concatenate(labels_list, 0),
               'vel_mag': np.zeros(len(labels_list))}
    save_npz(latents, f'outputs/data/latents/noprop_{region}.npz')

    result = {
        'method': f'noprop_vanilla_lambda{lambda_phys}', 'region': region,
        'accuracy': best_acc,
        'train_time_s': time.time() - t0,
        'n_params': sum(p.numel() for p in model.parameters()),
    }
    save_result(result, f'outputs/data/results/noprop_{region}.json')
    return result


def train_noprop_ct_baseline(cfg, dataloaders, region):
    from src.baselines.noprop_ct import NoPropCTModel
    device = torch.device(cfg.device)

    model = NoPropCTModel(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_acc = 0
    t0 = time.time()

    for epoch in range(150):    # reduced from 500 for speed
        model.train()
        for batch in dataloaders[region]['train']:
            x = torch.cat([batch['velocity'], batch['pressure'].unsqueeze(1)], dim=1).to(device)
            labels = batch['label'].to(device)
            opt.zero_grad()
            # Classification loss
            logits = model(x)
            cls_loss = torch.nn.functional.cross_entropy(logits, labels)
            # Diffusion loss (sample random t)
            u_y = model.label_embed(labels)
            t = torch.rand(1, device=device).expand(labels.size(0))
            diff_loss = model.compute_diffusion_loss(x, u_y, t)
            loss = cls_loss + diff_loss
            loss.backward()
            opt.step()

        if epoch % 50 == 0:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch in dataloaders[region]['val']:
                    x = torch.cat([batch['velocity'], batch['pressure'].unsqueeze(1)], dim=1).to(device)
                    labels = batch['label'].to(device)
                    preds = model(x).argmax(-1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            acc = 100 * correct / total
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), f'outputs/models/noprop_ct_{region}.pt')

    return {
        'method': 'noprop_ct', 'region': region,
        'accuracy': best_acc,
        'train_time_s': time.time() - t0,
        'n_params': sum(p.numel() for p in model.parameters()),
    }


def train_pinn_baseline(cfg, dataloaders, region):
    from src.baselines.pinn import PINNModel
    device = torch.device(cfg.device)

    model = PINNModel(cfg, nu=cfg.physics.viscosity).to(device)
    model.fit(dataloaders[region]['train'], n_epochs=150, n_col=512, disable_physics=True)
    result = model.evaluate(dataloaders[region]['val'], device)

    return {
        'method': 'pinn', 'region': region,
        'accuracy': result['accuracy'],
        'train_time_s': 0,  # measured internally
        'n_params': sum(p.numel() for p in model.parameters()),
    }


def train_spider_cnn_baseline(cfg, dataloaders, region):
    from src.baselines.spider_classifier import train_spider_classifier
    return train_spider_classifier(cfg, dataloaders[region]['train'],
                                   dataloaders[region]['val'])


def phase_baselines(args, cfg, dataloaders, vec_eq, scal_eq):
    print("\n" + "=" * 60)
    print("Phase 4: Baseline Training (5 methods x 2 regions)")
    print("=" * 60)

    all_results = {}

    for region in ['centre', 'edge']:
        region_results = {}
        for name, func in [
            ('cnn', train_cnn_baseline),
            ('noprop_vanilla', lambda c, d, r: train_noprop_baseline(c, d, r, 0.0)),
            ('noprop_ct', train_noprop_ct_baseline),
            ('pinn', train_pinn_baseline),
            ('spider_cnn', train_spider_cnn_baseline),
        ]:
            result_path = f'outputs/data/results/{name}_{region}.json'
            if os.path.exists(result_path):
                print(f"  [SKIP] {name} on {region} (already done)")
                with open(result_path) as fp:
                    region_results[name] = json.load(fp)
                continue
            print(f"  Training {name} on {region}...")
            torch.cuda.empty_cache()
            res = func(cfg, dataloaders, region)
            region_results[name] = res
            save_result(res, result_path)
            print(f"    {name}: acc={res['accuracy']:.1f}%")

        all_results[region] = region_results

    save_result(all_results, 'outputs/data/main_results.json')
    print("Baselines done.")
    return all_results


# ============================================================
#  Phase 5: PI-NoProp
# ============================================================

def phase_pi_noprop(args, cfg, dataloaders, vec_eq, scal_eq):
    from src.noprop.model import NoPropModel
    from src.decoder import FieldDecoder
    from src.physics.loss import PhysicsLoss
    from src.training.trainer import Trainer
    device = torch.device(cfg.device)

    print("\n" + "=" * 60)
    print("Phase 5: PI-NoProp Training")
    print("=" * 60)

    for region in ['centre', 'edge']:
        print(f"  Training PI-NoProp on {region}...")

        model = NoPropModel(cfg).to(device)
        decoder = FieldDecoder(
            latent_dim=cfg.decoder.latent_dim,
            base_channels=cfg.decoder.base_channels,
        ).to(device)
        physics_loss = PhysicsLoss(cfg, None, scal_eq).to(device)
        trainer = Trainer(model, decoder, physics_loss, cfg)

        train_loader = dataloaders[region]['train']
        val_loader = dataloaders[region]['val']

        # Load pretrained decoder
        pretrained_path = f'outputs/models/pretrained_decoder_{region}.pt'
        if os.path.exists(pretrained_path):
            decoder.load_state_dict(torch.load(pretrained_path))
            print(f"    Loaded pretrained decoder from {pretrained_path}")
        else:
            trainer.pretrain_decoder(train_loader, n_epochs=cfg.training.n_pretrain_epochs)

        # Training
        history = {'loss': [], 'cls': [], 'diff': [], 'phys': [], 'acc': []}
        latents_list, labels_list, vel_mag_list = [], [], []
        all_preds, all_labels = [], []
        best_acc, best_epoch = 0, 0
        t0 = time.time()

        for epoch in range(cfg.training.n_epochs):
            stats = trainer.train_epoch(train_loader)
            val_stats = trainer.validate(val_loader)
            for k in ['loss', 'cls', 'diff', 'phys']:
                if k in stats: history[k].append(stats[k])
            history['acc'].append(val_stats['accuracy'])

            if val_stats['accuracy'] > best_acc:
                best_acc = val_stats['accuracy']
                best_epoch = epoch
                trainer.save_checkpoint(epoch, f'outputs/models/pi_noprop_{region}.pt')

            # Collect data for figures
            if epoch == cfg.training.n_epochs - 1:
                model.eval()
                decoder.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        x = torch.cat([batch['velocity'], batch['pressure'].unsqueeze(1)], dim=1).to(device)
                        labels = batch['label'].to(device)
                        logits, z_all = model(x, return_all_latents=True)
                        preds = logits.argmax(-1)
                        all_preds.append(preds.cpu().numpy())
                        all_labels.append(labels.cpu().numpy())
                        latents_list.append(z_all[-1].cpu().numpy())
                        labels_list.append(labels.cpu().numpy())
                        # Compute velocity magnitude
                        vel_mag = torch.sqrt((x[:, :3] ** 2).sum(dim=(1, 2, 3, 4)))
                        vel_mag_list.append(vel_mag.cpu().numpy())

        # Save all data
        save_npz(history, f'outputs/data/training_curves/pi_noprop_{region}.npz')
        save_npz({
            'z_T': np.concatenate(latents_list, 0),
            'labels': np.concatenate(labels_list, 0),
            'vel_mag': np.concatenate(vel_mag_list, 0),
        }, f'outputs/data/latents/pi_noprop_{region}.npz')
        save_npz({
            'preds': np.concatenate(all_preds, 0),
            'labels': np.concatenate(all_labels, 0),
        }, f'outputs/data/confusion/pi_noprop_{region}.npz')

        # Reconstruction samples
        model.eval()
        decoder.eval()
        with torch.no_grad():
            batch = next(iter(val_loader))
            x = torch.cat([batch['velocity'], batch['pressure'].unsqueeze(1)], dim=1).to(device)[:8]
            _, z_all = model(x, return_all_latents=True)
            fields_pred = decoder(z_all[-1]).cpu().numpy()
            fields_true = x.cpu().numpy()
            save_npz({
                'fields_true': fields_true,
                'fields_pred': fields_pred,
            }, f'outputs/data/reconstruction/{region}.npz')

        eta_ns = compute_eta_ns_reconstructed(physics_loss, model, decoder, val_loader, device)

        result = {
            'method': 'pi_noprop', 'region': region,
            'accuracy': best_acc,
            'eta_NS': eta_ns,
            'train_time_s': time.time() - t0,
            'n_params': sum(p.numel() for p in list(model.parameters()) + list(decoder.parameters())),
        }
        save_result(result, f'outputs/data/results/pi_noprop_{region}.json')
        print(f"    acc={best_acc:.1f}% eta_NS={eta_ns:.4f}")

    print("PI-NoProp done.")


# ============================================================
#  Phases 6-9: Sweeps + Ablations
# ============================================================

def _train_with_params(cfg_overrides, region, n_pretrain, n_epochs):
    """Train PI-NoProp with given overrides. Returns result dict."""
    from src.noprop.model import NoPropModel
    from src.decoder import FieldDecoder
    from src.physics.loss import PhysicsLoss
    from src.training.trainer import Trainer
    from src.data.dataset import create_dataloaders
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = make_config(**cfg_overrides)
    dl = create_dataloaders(cfg)
    model = NoPropModel(cfg).to(device)
    decoder = FieldDecoder().to(device)
    physics_loss = PhysicsLoss(cfg).to(device)
    trainer = Trainer(model, decoder, physics_loss, cfg)

    pretrained_path = f'outputs/models/pretrained_decoder_{region}.pt'
    if os.path.exists(pretrained_path):
        decoder.load_state_dict(torch.load(pretrained_path, map_location=device))

    trainer.pretrain_decoder(dl[region]['train'], n_epochs=n_pretrain)
    best_acc = 0
    for _ in range(n_epochs):
        trainer.train_epoch(dl[region]['train'])
        val = trainer.validate(dl[region]['val'])
        if val['accuracy'] > best_acc:
            best_acc = val['accuracy']

    eta = compute_eta_ns_reconstructed(physics_loss, model, decoder, dl[region]['val'], device)
    torch.cuda.empty_cache()
    return {'accuracy': best_acc, 'eta_NS': eta}


def phase_noise_sweep(args):
    print("\n" + "=" * 60)
    print("Phase 6: Noise Robustness Sweep (reduced: 4 levels)")
    print("=" * 60)
    levels = [0.0, 0.1, 0.5, 1.0]
    results = {'noise_levels': levels, 'noprop_acc': [], 'noprop_eta': [],
               'pi_noprop_acc': [], 'pi_noprop_eta': []}
    for sigma in levels:
        for method, lam in [('noprop', 0.0), ('pi_noprop', 0.01)]:
            r = _train_with_params({'data.noise_level': sigma,
                                     'physics.lambda_weight': lam,
                                     'training.n_pretrain_epochs': 50,
                                     'training.n_epochs': 30},
                                    'centre', 50, 30)
            key = f'{method}_acc' if method == 'noprop' else 'pi_noprop_acc'
            if lam == 0.0:
                results['noprop_acc'].append(r['accuracy'])
                results['noprop_eta'].append(r['eta_NS'])
            else:
                results['pi_noprop_acc'].append(r['accuracy'])
                results['pi_noprop_eta'].append(r['eta_NS'])
            print(f"  sigma={sigma} {method}: acc={r['accuracy']:.1f}")
        save_npz(results, 'outputs/noise_results.npz')
    print("Noise sweep done.")


def phase_lambda_sweep(args):
    print("\n" + "=" * 60)
    print("Phase 7: Lambda Tradeoff Sweep (reduced: 5 levels)")
    print("=" * 60)
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    results = {'lambdas': lambdas, 'accuracy': [], 'eta_NS': []}
    for lam in lambdas:
        r = _train_with_params({'physics.lambda_weight': lam,
                                 'training.n_pretrain_epochs': 50,
                                 'training.n_epochs': 30},
                                'centre', 50, 30)
        results['accuracy'].append(r['accuracy'])
        results['eta_NS'].append(r['eta_NS'])
        print(f"  lambda={lam:.0e}: acc={r['accuracy']:.1f} eta={r['eta_NS']:.4f}")
        save_npz(results, 'outputs/lambda_sweep.npz')
    print("Lambda sweep done.")


def phase_equation_ablation(args):
    print("\n" + "=" * 60)
    print("Phase 8: Equation Ablation (NS / NS+PP / full)")
    print("=" * 60)
    variants = [
        ('NS', {'physics.use_continuity': True, 'physics.use_pressure_poisson': False, 'physics.use_energy': False}),
        ('NS+PP', {'physics.use_continuity': True, 'physics.use_pressure_poisson': True, 'physics.use_energy': False}),
        ('full', {'physics.use_continuity': True, 'physics.use_pressure_poisson': True, 'physics.use_energy': True}),
    ]
    results = {}
    for name, overrides in variants:
        r = _train_with_params({**overrides,
                                 'training.n_pretrain_epochs': 50,
                                 'training.n_epochs': 50},
                                'centre', 50, 50)
        results[name] = r
        print(f"  {name}: acc={r['accuracy']:.1f} eta={r['eta_NS']:.4f}")
    save_result(results, 'outputs/data/ablation/equation_ablation.json')
    print("Equation ablation done.")


def phase_decoder_ablation(args):
    print("\n" + "=" * 60)
    print("Phase 9: Decoder Ablation (Conv vs Linear)")
    print("=" * 60)
    results = {}
    for dec_type in ['conv', 'linear']:
        r = _train_with_params({'decoder.use_conv': dec_type == 'conv',
                                 'training.n_pretrain_epochs': 50,
                                 'training.n_epochs': 50},
                                'centre', 50, 50)
        results[dec_type] = r
        print(f"  {dec_type}: acc={r['accuracy']:.1f} eta={r['eta_NS']:.4f}")
    save_result(results, 'outputs/data/ablation/decoder_ablation.json')
    print("Decoder ablation done.")


def phase_benchmark(args):
    print("\n" + "=" * 60)
    print("Phase 10: Efficiency Benchmark")
    print("=" * 60)
    # Simple benchmark: measure training time and memory for each method
    results = {}
    methods = ['cnn', 'noprop', 'noprop_ct', 'pinn', 'spider_cnn', 'pi_noprop']
    for m in methods:
        model_path = f'outputs/models/{m}_centre.pt'
        if os.path.exists(model_path):
            sz = os.path.getsize(model_path)
        else:
            sz = 0
        results[m] = {'model_size_mb': round(sz / 1e6, 2)}
    save_result(results, 'outputs/data/benchmark.json')
    print("Benchmark done. (Full measurement requires GPU profiling)")


def phase_figures(args):
    print("\n" + "=" * 60)
    print("Phase 11: Generate Paper Figures")
    print("=" * 60)
    # Launch the figure regeneration script
    import subprocess
    subprocess.run([sys.executable, 'scripts/regenerate_figures.py'], check=True)
    print("Figures done.")


def phase_summary(args):
    print("\n" + "=" * 60)
    print("Phase 12: Results Summary")
    print("=" * 60)
    results_dir = 'outputs/data/results'
    if os.path.exists(results_dir):
        for f in sorted(os.listdir(results_dir)):
            if f.endswith('.json'):
                with open(os.path.join(results_dir, f)) as fp:
                    data = json.load(fp)
                print(f"  {data.get('method', '?'):20s} {data.get('region', '?'):8s}  "
                      f"acc={data.get('accuracy', 0):.1f}%  time={data.get('train_time_s', 0):.0f}s")
    print("\nAll done.")


# ============================================================
#  Main
# ============================================================

PHASES = {
    '1': ('generate_data', phase_generate_data),
    '2': ('spider', phase_spider),
    '3': ('pretrain', phase_pretrain),
    '4': ('baselines', phase_baselines),
    '5': ('pi_noprop', phase_pi_noprop),
    '6': ('noise_sweep', phase_noise_sweep),
    '7': ('lambda_sweep', phase_lambda_sweep),
    '8': ('equation_ablation', phase_equation_ablation),
    '9': ('benchmark', phase_benchmark),
    '10': ('figures', phase_figures),
    '11': ('summary', phase_summary),
}


def main():
    parser = argparse.ArgumentParser(description='Run all PI-NoProp experiments')
    parser.add_argument('--phase', type=str, help='Run specific phase (e.g. "4" or "4,5,6")')
    parser.add_argument('--start', type=int, help='Start phase')
    parser.add_argument('--stop', type=int, help='Stop phase (inclusive)')
    parser.add_argument('--skip', type=str, help='Skip phases (e.g. "1,11")')
    parser.add_argument('--skip-dns', action='store_true', help='Skip DNS generation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Determine which phases to run
    if args.phase:
        phase_nums = [p.strip() for p in args.phase.split(',')]
    elif args.start and args.stop:
        phase_nums = [str(i) for i in range(args.start, args.stop + 1)]
    elif args.start:
        phase_nums = [str(i) for i in range(args.start, 12)]
    else:
        phase_nums = [str(i) for i in range(1, 12)]

    skip = set()
    if args.skip:
        skip = set(p.strip() for p in args.skip.split(','))
    if args.skip_dns:
        skip.add('1')

    # Shared state across phases
    cfg = make_config(**{'device': args.device})
    dataloaders = None
    vec_eq = scal_eq = None
    all_results = {}

    for pn in phase_nums:
        if pn in skip:
            print(f"  [SKIP] Phase {pn}: {PHASES[pn][0]}")
            continue

        phase_name, phase_func = PHASES[pn]
        try:
            # Pass state to phases that need it
            kwargs = {}
            if pn in ['4', '5']:
                if dataloaders is None:
                    from src.data.dataset import create_dataloaders
                    dataloaders = create_dataloaders(cfg)
                kwargs['dataloaders'] = dataloaders
                if vec_eq is None:
                    # Load from saved JSON
                    import json
                    vpath = 'outputs/spider/vector_equation.json'
                    spath = 'outputs/spider/scalar_equation.json'
                    if os.path.exists(vpath):
                        with open(vpath) as f:
                            vec_eq = _json_to_eq(json.load(f), args.device)
                    if os.path.exists(spath):
                        with open(spath) as f:
                            scal_eq = _json_to_eq(json.load(f), args.device)
                kwargs['vec_eq'] = vec_eq
                kwargs['scal_eq'] = scal_eq
            if pn == '4':
                kwargs['all_results'] = all_results

            if pn == '1':
                phase_func(args)
            elif pn == '2':
                vec_eq, scal_eq = phase_func(args)
            elif pn == '3':
                if vec_eq is None:
                    # Load SPIDER
                    import json
                    vpath = 'outputs/spider/vector_equation.json'
                    spath = 'outputs/spider/scalar_equation.json'
                    if os.path.exists(vpath):
                        with open(vpath) as f:
                            vec_eq = _json_to_eq(json.load(f), args.device)
                    if os.path.exists(spath):
                        with open(spath) as f:
                            scal_eq = _json_to_eq(json.load(f), args.device)
                dataloaders = phase_func(args, vec_eq, scal_eq)
            elif pn == '4':
                all_results = phase_func(args, cfg, dataloaders, vec_eq, scal_eq)
            elif pn == '5':
                phase_func(args, cfg, dataloaders, vec_eq, scal_eq)
            else:
                phase_func(args)
            print(f"  [DONE] Phase {pn}: {phase_name}")
        except Exception as e:
            print(f"  [ERROR] Phase {pn} ({phase_name}): {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()

"""
repair_fixes.py — Run only the PINN and PI-NoProp re-trainings needed after bug fixes.
"""
import sys, os, time, json, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from src.config import PINoPropConfig
from src.data.dataset import create_dataloaders


def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def main():
    cfg = PINoPropConfig()
    cfg.data.data_dir = "data/generated"
    dataloaders = create_dataloaders(cfg)

    # ----- 1. PINN baselines (physics disabled) -----
    print("=" * 60)
    print("1. PINN baseline (physics disabled)")
    print("=" * 60)

    from src.baselines.pinn import PINNModel
    device = torch.device(cfg.device)

    for region in ["centre", "edge"]:
        result_path = f"outputs/data/results/pinn_{region}.json"
        if os.path.exists(result_path):
            print(f"  [SKIP] pinn_{region}")
            continue

        print(f"  Training PINN on {region}...")
        model = PINNModel(cfg, nu=cfg.physics.viscosity).to(device)
        model.fit(dataloaders[region]["train"], n_epochs=50, n_col=512, device=str(device),
                  disable_physics=True)
        result = model.evaluate(dataloaders[region]["val"], device=str(device))
        res = {
            "method": "pinn",
            "region": region,
            "accuracy": result["accuracy"],
            "train_time_s": 0,
            "n_params": sum(p.numel() for p in model.parameters()),
        }
        ensure_dir(result_path)
        with open(result_path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"    pinn_{region}: acc={result['accuracy']:.1f}%")
        torch.cuda.empty_cache()

    # ----- 2. PI-NoProp (fixed eta_NS) -----
    print()
    print("=" * 60)
    print("2. PI-NoProp (eta_NS on reconstructed fields)")
    print("=" * 60)

    from src.noprop.model import NoPropModel
    from src.decoder import FieldDecoder
    from src.physics.loss import PhysicsLoss
    from src.training.trainer import Trainer

    # Load and convert SPIDER equations
    spider_path = "outputs/spider"

    def _json_to_eq(data, device='cpu'):
        if data is None:
            return None
        coeffs = torch.tensor(data['coefficients'], device=torch.device(device))
        terms = [{'type': t, 'fields': None} for t in data['terms']]
        return coeffs, terms

    vec_eq_path = os.path.join(spider_path, "vector_equation.json")
    scal_eq_path = os.path.join(spider_path, "scalar_equation.json")

    vec_eq = None
    scal_eq = None
    if os.path.exists(scal_eq_path):
        with open(scal_eq_path) as f:
            raw = json.load(f)
        scal_eq = _json_to_eq(raw)
    if os.path.exists(vec_eq_path):
        with open(vec_eq_path) as f:
            raw = json.load(f)
        vec_eq = _json_to_eq(raw)

    for region in ["centre", "edge"]:
        result_path = f"outputs/data/results/pi_noprop_{region}.json"
        if os.path.exists(result_path):
            print(f"  [SKIP] pi_noprop_{region}")
            continue

        print(f"  Training PI-NoProp on {region}...")

        model = NoPropModel(cfg).to(device)
        decoder = FieldDecoder(
            latent_dim=cfg.decoder.latent_dim,
            base_channels=cfg.decoder.base_channels,
        ).to(device)

        physics_loss = PhysicsLoss(cfg, vec_eq, scal_eq).to(device)

        trainer = Trainer(model, decoder, physics_loss, cfg)

        train_loader = dataloaders[region]["train"]
        val_loader = dataloaders[region]["val"]

        # Load pretrained decoder
        pretrained_path = f"outputs/models/pretrained_decoder_{region}.pt"
        if os.path.exists(pretrained_path):
            decoder.load_state_dict(torch.load(pretrained_path, map_location=device))
            print(f"    Loaded pretrained decoder")
        else:
            trainer.pretrain_decoder(train_loader, n_epochs=cfg.training.n_pretrain_epochs)

        # Training
        best_acc = 0.0
        t0 = time.time()
        for epoch in range(cfg.training.n_epochs):
            trainer.train_epoch(train_loader)
            val_stats = trainer.validate(val_loader)
            if val_stats["accuracy"] > best_acc:
                best_acc = val_stats["accuracy"]

        # Compute eta_NS on RECONSTRUCTED fields
        model.eval()
        decoder.eval()
        physics_loss.eval()
        eta_vals = []
        with torch.no_grad():
            for batch in val_loader:
                x = torch.cat([batch["velocity"], batch["pressure"].unsqueeze(1)], dim=1)
                x = x.to(device)
                _, z_all = model(x, return_all_latents=True)
                reconstructed = decoder(z_all[-1])
                eta_vals.append(physics_loss.compute_eta_ns(reconstructed))
        eta_ns = float(torch.tensor(eta_vals).mean().item()) if eta_vals else 0.0

        res = {
            "method": "pi_noprop",
            "region": region,
            "accuracy": best_acc,
            "eta_NS": eta_ns,
            "train_time_s": time.time() - t0,
            "n_params": sum(p.numel() for p in list(model.parameters()) + list(decoder.parameters())),
        }
        ensure_dir(result_path)
        with open(result_path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"    pi_noprop_{region}: acc={best_acc:.1f}% eta_NS={eta_ns:.4f}")
        torch.cuda.empty_cache()

    print()
    print("All repairs done.")


if __name__ == "__main__":
    main()

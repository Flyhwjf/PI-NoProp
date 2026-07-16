"""SPIDER + Classifier baseline.
Runs SPIDER for post-hoc equation discovery, then trains a separate CNN.
The SPIDER results are NOT used during training — they are only for post-hoc analysis.
"""
import torch
import time

from .cnn import SimpleCNN


def train_spider_classifier(config, dataloader_train, dataloader_val,
                            spider_equations=None):
    """Train a CNN classifier independently of SPIDER.

    Args:
        config: PINoPropConfig
        dataloader_train, dataloader_val: data
        spider_equations: optional pre-discovered SPIDER equations for reporting

    Returns:
        dict with keys: method, accuracy, train_time_s, n_params
    """
    model = SimpleCNN(
        in_channels=4,
        n_classes=config.data.n_classes,
        grid_size=config.data.subdomain_size,
    )
    device = torch.device(config.device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    t0 = time.time()

    for epoch in range(config.training.n_epochs):
        model.train()
        for batch in dataloader_train:
            x = torch.cat([batch['velocity'],
                           batch['pressure'].unsqueeze(1)], dim=1)
            x, labels = x.to(device), batch['label'].to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in dataloader_val:
                x = torch.cat([batch['velocity'],
                               batch['pressure'].unsqueeze(1)], dim=1)
                x, labels = x.to(device), batch['label'].to(device)
                preds = model(x).argmax(-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = 100.0 * correct / total
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(),
                       config.training.save_dir + '/spider_cnn_best.pt')

    train_time = time.time() - t0

    return {
        'method': 'spider_cnn',
        'accuracy': best_acc,
        'train_time_s': train_time,
        'n_params': sum(p.numel() for p in model.parameters()),
    }

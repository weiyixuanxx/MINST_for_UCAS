import argparse
import os
import random
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from dataset import create_dataloaders
from model import MNISTCNN


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct


def run_epoch(model, loader, criterion, device, optimizer=None, max_batches=0):
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += accuracy(logits, labels)
            total_samples += batch_size

            if max_batches > 0 and (batch_idx + 1) >= max_batches:
                break

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    return avg_loss, avg_acc


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError("YAML config must be a mapping (key-value pairs).")

    # Accept both snake_case and kebab-case keys from YAML.
    normalized = {str(key).replace("-", "_"): value for key, value in config.items()}
    return normalized


def build_parser():
    parser = argparse.ArgumentParser(description="Train MNIST digit classifier.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/mnist_cnn.pt",
    )
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    return parser


def parse_args():
    parser = build_parser()
    pre_args, _ = parser.parse_known_args()

    if pre_args.config:
        yaml_config = load_config(pre_args.config)
        valid_keys = {action.dest for action in parser._actions}
        invalid_keys = sorted(k for k in yaml_config if k not in valid_keys)
        if invalid_keys:
            raise ValueError("Unknown keys in config: " + ", ".join(invalid_keys))
        parser.set_defaults(**yaml_config)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.config:
        print(f"Loaded config: {args.config}")
    set_seed(args.seed)

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    pin_memory = device.type == "cuda"

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
        pin_memory=pin_memory,
    )

    model = MNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_acc = 0.0
    checkpoint_dir = os.path.dirname(args.checkpoint)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
        )

        val_loss, val_acc = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            max_batches=args.max_eval_batches,
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "args": vars(args),
                },
                args.checkpoint,
            )

    test_loss, test_acc = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        max_batches=args.max_eval_batches,
    )
    print(f"Final test_loss={test_loss:.4f} test_acc={test_acc:.4f}")
    print(f"Best checkpoint saved to: {args.checkpoint}")


if __name__ == "__main__":
    main()

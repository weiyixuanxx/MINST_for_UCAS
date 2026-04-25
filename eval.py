import argparse

import torch
import torch.nn as nn

from dataset import create_dataloaders
from model import MNISTCNN


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    return avg_loss, avg_acc


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MNIST digit classifier.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/mnist_cnn.pt",
    )
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    pin_memory = device.type == "cuda"

    _, _, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=0.1,
        seed=42,
        pin_memory=pin_memory,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = MNISTCNN().to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"checkpoint: {args.checkpoint}")
    print(f"test_loss={test_loss:.4f}")
    print(f"test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()

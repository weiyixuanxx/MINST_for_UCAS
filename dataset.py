import torch
from torch.utils.data import DataLoader, random_split

from data import load_mnist


def create_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    num_workers: int = 2,
    val_split: float = 0.1,
    seed: int = 42,
    pin_memory: bool = True,
):
    if not 0 <= val_split < 1:
        raise ValueError("val_split must be in [0, 1).")

    train_dataset, test_dataset = load_mnist(data_dir=data_dir, download=True)

    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader

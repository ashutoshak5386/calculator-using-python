from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return train and eval transforms.

    For medical imaging, keep augmentations modest by default.
    """
    train_t = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_t = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_t, eval_t


def create_dataloaders(
    data_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """Create train and val dataloaders using ImageFolder splits."""
    train_t, eval_t = build_transforms(image_size=image_size)

    train_ds = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_t)
    val_ds = datasets.ImageFolder(root=f"{data_dir}/val", transform=eval_t)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader, train_ds.classes


def create_eval_loader(
    data_dir: str,
    split: str,
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """Create an evaluation dataloader for a given split (val or test)."""
    _, eval_t = build_transforms(image_size=image_size)
    ds = datasets.ImageFolder(root=f"{data_dir}/{split}", transform=eval_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return loader, ds.classes

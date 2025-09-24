from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(directory_path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(directory_path, exist_ok=True)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(
    file_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch_index: int,
    best_val_loss: float,
    class_names: List[str],
    args: Dict[str, Any],
) -> None:
    """Save a training checkpoint to disk."""
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch_index,
        "best_val_loss": best_val_loss,
        "class_names": class_names,
        "args": args,
        "version": "0.1.0",
    }
    torch.save(checkpoint, file_path)


def load_checkpoint(file_path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
    """Load a training checkpoint from disk."""
    return torch.load(file_path, map_location=map_location)


def write_lines(file_path: str, lines: List[str]) -> None:
    with open(file_path, "w", encoding="utf-8") as file:
        for line in lines:
            file.write(f"{line}\n")


def read_lines(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file.readlines() if line.strip()]

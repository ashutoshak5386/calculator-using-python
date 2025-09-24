from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import create_eval_loader
from .model import build_resnet18_classifier
from .utils import ensure_dir, get_device, load_checkpoint, read_lines


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []
    for images, targets in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(targets.numpy())
    return np.array(all_targets), np.array(all_preds)


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes, ylabel="True label", xlabel="Predicted label", title="Confusion Matrix")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model on a split")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    device = get_device()
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")

    # Load classes
    classes_path = os.path.join(os.path.dirname(args.checkpoint), "classes.txt")
    if os.path.exists(classes_path):
        class_names = read_lines(classes_path)
    else:
        # Fallback from checkpoint if available
        class_names = ckpt.get("class_names", [])

    model, img_size = build_resnet18_classifier(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    loader, ds_classes = create_eval_loader(
        data_dir=args.data_dir, split=args.split, image_size=img_size, batch_size=64
    )
    assert ds_classes == class_names, "Dataset classes differ from training classes. Check data folders."

    y_true, y_pred = evaluate_model(model, loader, device)

    acc = accuracy_score(y_true, y_pred)
    print(f"accuracy={acc:.4f}")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, os.path.join(args.output_dir, "confusion_matrix.png"))


if __name__ == "__main__":
    main()

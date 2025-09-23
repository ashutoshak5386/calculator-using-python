from __future__ import annotations

import argparse
import glob
import os
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms

from .model import build_resnet18_classifier
from .utils import get_device, load_checkpoint, read_lines


def build_eval_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def predict_image(model: torch.nn.Module, image_path: str, transform: transforms.Compose, device: torch.device) -> Tuple[int, float]:
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)
    conf, pred = torch.max(probs, dim=1)
    return int(pred.item()), float(conf.item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on image or folder")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str)
    group.add_argument("--folder", type=str)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    device = get_device()
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")

    classes_path = os.path.join(os.path.dirname(args.checkpoint), "classes.txt")
    class_names = read_lines(classes_path) if os.path.exists(classes_path) else ckpt.get("class_names", [])

    model, img_size = build_resnet18_classifier(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    transform = build_eval_transform(img_size)

    image_paths: List[str] = []
    if args.image:
        image_paths = [args.image]
    else:
        exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
        for ext in exts:
            image_paths.extend(glob.glob(os.path.join(args.folder, ext)))
        image_paths.sort()

    for path in image_paths:
        pred_idx, conf = predict_image(model, path, transform, device)
        label = class_names[pred_idx] if 0 <= pred_idx < len(class_names) else str(pred_idx)
        print(f"{path}: {label} ({conf:.3f})")


if __name__ == "__main__":
    main()

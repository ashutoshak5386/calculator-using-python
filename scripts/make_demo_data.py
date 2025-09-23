from __future__ import annotations

import os
import random
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def generate_cn_image(size: Tuple[int, int]) -> Image.Image:
    # Control image: smooth grayscale gradient with slight blur
    width, height = size
    x = np.linspace(0, 255, width, dtype=np.uint8)
    gradient = np.tile(x, (height, 1))
    img = np.stack([gradient, gradient, gradient], axis=2)
    pil = Image.fromarray(img)
    pil = pil.filter(ImageFilter.GaussianBlur(radius=1.0))
    return pil


def generate_ad_image(size: Tuple[int, int]) -> Image.Image:
    # AD-like: more texture/noise with a colored ring pattern
    width, height = size
    noise = np.random.normal(loc=128, scale=40, size=(height, width, 3)).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(noise)
    draw = ImageDraw.Draw(pil)
    cx, cy = width // 2, height // 2
    for r in range(min(cx, cy), 10, -20):
        color = (random.randint(100, 255), random.randint(0, 155), random.randint(0, 155))
        bbox = (cx - r, cy - r, cx + r, cy + r)
        draw.ellipse(bbox, outline=color, width=3)
    pil = pil.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    return pil


def main() -> None:
    root = "/workspace/data_demo"
    img_size = (224, 224)
    splits = {"train": 24, "val": 8}
    classes = ["CN", "AD"]

    for split, count in splits.items():
        for cls in classes:
            out_dir = os.path.join(root, split, cls)
            ensure_dir(out_dir)
            for i in range(count):
                if cls == "CN":
                    img = generate_cn_image(img_size)
                else:
                    img = generate_ad_image(img_size)
                img.save(os.path.join(out_dir, f"{cls.lower()}_{i:03d}.png"))

    print(f"Created demo dataset at {root}")


if __name__ == "__main__":
    main()

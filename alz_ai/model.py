from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models


def build_resnet18_classifier(num_classes: int, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """Create a ResNet18-based classifier for transfer learning.

    Returns the model and the input image size requirement (224).
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )

    return model, 224

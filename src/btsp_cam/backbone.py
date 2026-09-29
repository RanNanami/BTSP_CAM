"""Frozen feature interfaces, including the PILOT ViT-B/16 checkpoint path."""

from __future__ import annotations

import numpy as np


class IdentityBackbone:
    """Treat a two-dimensional array as an already-frozen feature batch."""

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        features = np.asarray(inputs, dtype=np.float64)
        if features.ndim != 2:
            raise ValueError("features must have shape (batch, feature_dim)")
        return features


PILOT_TIMM_0612_MODEL = "vit_base_patch16_224_in21k"
TIMM_CURRENT_EQUIVALENT = "vit_base_patch16_224.augreg_in21k"


class PilotFrozenViT:
    """Frozen PILOT ViT-B/16 ImageNet-21K feature extractor.

    PILOT pins timm 0.6.12 and uses ``vit_base_patch16_224_in21k``.
    Modern timm renamed the same Google AugReg checkpoint to
    ``vit_base_patch16_224.augreg_in21k``. Both resolve to the identical
    ``B_16-i21k-300ep...npz`` source URL.
    """

    def __init__(self, device: str = "cpu", pretrained: bool = True):
        import timm
        import torch

        self.timm_version = timm.__version__
        self.model_name = TIMM_CURRENT_EQUIVALENT
        self.device = torch.device(device)
        self.model = timm.create_model(
            self.model_name, pretrained=pretrained, num_classes=0
        ).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.output_dim = int(self.model.num_features)
        self.pretrained_cfg = dict(self.model.pretrained_cfg)
        if self.output_dim != 768:
            raise RuntimeError(f"unexpected PILOT ViT output dimension: {self.output_dim}")
        if pretrained and not self.pretrained_cfg:
            raise RuntimeError("pretrained checkpoint metadata is missing")

    def extract_loader(self, loader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch

        feature_batches, label_batches, index_batches = [], [], []
        with torch.inference_mode():
            for indices, images, labels in loader:
                if not torch.isfinite(images).all():
                    raise ValueError("backbone input contains NaN or Inf")
                features = self.model(images.to(self.device))
                if not torch.isfinite(features).all():
                    raise FloatingPointError("backbone produced NaN or Inf")
                feature_batches.append(features.cpu().numpy())
                label_batches.append(labels.numpy())
                index_batches.append(indices.numpy())
        return (
            np.concatenate(feature_batches),
            np.concatenate(label_batches),
            np.concatenate(index_batches),
        )

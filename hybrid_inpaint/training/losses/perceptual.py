"""VGG16 perceptual loss (``relu3_3`` and ``relu4_3``).

The extractor is torchvision's ImageNet-pretrained VGG16, frozen and kept in
eval mode.  Inputs are ``[0,1]`` RGB; the module normalises them with the
ImageNet mean/std before the forward pass.  The returned value is the sum of
the two L1 feature distances; the caller applies the 0.1 weight from config.

A custom ``feature_extractor`` can be injected for tests; it must accept a
BCHW tensor and return ``(relu3_3_features, relu4_3_features)``.
"""

from __future__ import annotations

import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VGG16_WEIGHTS_ID = "torchvision:vgg16:IMAGENET1K_V1"
RELU3_3_INDEX = 16  # features[:16] ends at relu3_3
RELU4_3_INDEX = 23  # features[:23] ends at relu4_3


class _VGG16FeatureExtractor(nn.Module):
    """Frozen VGG16 producing relu3_3 and relu4_3 features."""

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.stage3 = features[:RELU3_3_INDEX]
        self.stage4 = features[RELU3_3_INDEX:RELU4_3_INDEX]
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "_VGG16FeatureExtractor":
        # The frozen extractor must stay in eval mode even when the trainer
        # calls ``model.train()`` on its parent modules.
        return super().train(False)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stage3 = self.stage3(inputs)
        stage4 = self.stage4(stage3)
        return stage3, stage4


class PerceptualLoss(nn.Module):
    """Sum of L1 distances at VGG16 relu3_3 and relu4_3."""

    def __init__(self, feature_extractor: nn.Module | None = None) -> None:
        super().__init__()
        self.feature_extractor = (
            feature_extractor if feature_extractor is not None else _VGG16FeatureExtractor()
        )
        self.register_buffer(
            "mean",
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.weights_id = (
            VGG16_WEIGHTS_ID if feature_extractor is None else "custom:feature_extractor"
        )

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError("prediction and target shapes must match")
        if prediction.ndim != 4:
            raise ValueError("prediction and target must be BCHW tensors")

        prediction = (prediction.float() - self.mean) / self.std
        target = (target.float() - self.mean) / self.std
        pred3, pred4 = self.feature_extractor(prediction)
        target3, target4 = self.feature_extractor(target)
        return (pred3 - target3).abs().mean() + (pred4 - target4).abs().mean()

    def train(self, mode: bool = True) -> "PerceptualLoss":
        super().train(mode)
        self.feature_extractor.eval()
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad_(False)
        return self

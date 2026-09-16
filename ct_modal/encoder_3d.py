"""Native PyTorch encoder for fixed-shape L2-L4 ROI volumes."""

from __future__ import annotations

import torch
from torch import nn


class CTEncoder3D(nn.Module):
    """Map ``[B,1,D,H,W]`` ROI tensors to fixed-width embeddings."""

    def __init__(
        self,
        embedding_dim: int = 128,
        pretrained: bool = False,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        if pretrained:
            raise ValueError("CTEncoder3D has no pretrained weights")
        if (
            not isinstance(embedding_dim, int)
            or isinstance(embedding_dim, bool)
            or embedding_dim <= 0
        ):
            raise ValueError("embedding_dim must be a positive integer")
        if not 0.0 <= float(dropout) <= 1.0:
            raise ValueError("dropout must be in [0, 1]")
        self.embedding_dim = embedding_dim
        self.backbone = nn.Sequential(
            self._block(1, 16),
            nn.MaxPool3d(2),
            self._block(16, 32),
            nn.MaxPool3d(2),
            self._block(32, 64),
            nn.MaxPool3d(2),
            self._block(64, 128),
            nn.AdaptiveAvgPool3d(1),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(float(dropout)),
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )
        self._backbone_frozen = False
        self.set_backbone_trainable(not freeze_backbone)

    @staticmethod
    def _block(input_channels: int, output_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(
                input_channels, output_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm3d(output_channels),
            nn.GELU(),
        )

    @property
    def backbone_frozen(self) -> bool:
        return self._backbone_frozen

    def set_backbone_trainable(self, trainable: bool) -> None:
        if not isinstance(trainable, bool):
            raise ValueError("trainable must be bool")
        self._backbone_frozen = not trainable
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(trainable)
        self.backbone.train(self.training and trainable)

    def train(self, mode: bool = True) -> CTEncoder3D:
        super().train(mode)
        if self._backbone_frozen:
            self.backbone.eval()
        return self

    def forward(self, ct: torch.Tensor) -> torch.Tensor:
        single = self._validate_input(ct)
        if single:
            ct = ct.unsqueeze(0)
        embedding = self.projection(self.backbone(ct))
        if not torch.isfinite(embedding).all().item():
            raise ValueError("CT embedding contains non-finite values")
        return embedding.squeeze(0) if single else embedding

    def _validate_input(self, ct: torch.Tensor) -> bool:
        if not isinstance(ct, torch.Tensor):
            raise ValueError(f"ct must be a torch.Tensor, got {type(ct).__name__}")
        if ct.ndim not in (4, 5):
            raise ValueError(
                f"roi_3d ct must have shape [1,D,H,W] or [B,1,D,H,W], got {tuple(ct.shape)}"
            )
        channel_axis = 0 if ct.ndim == 4 else 1
        if ct.shape[channel_axis] != 1:
            raise ValueError("roi_3d ct must have one channel")
        if min(ct.shape[-3:]) < 8:
            raise ValueError("roi_3d spatial dimensions must each be at least 8")
        if not torch.is_floating_point(ct) or not torch.isfinite(ct).all().item():
            raise ValueError("roi_3d ct must contain finite floating-point values")
        if float(ct.min()) < 0.0 or float(ct.max()) > 1.0:
            raise ValueError("roi_3d ct must follow the preprocessing range [0, 1]")
        first_parameter = next(self.parameters())
        if ct.device != first_parameter.device:
            raise ValueError("ct and CTEncoder3D must be on the same device")
        return ct.ndim == 4

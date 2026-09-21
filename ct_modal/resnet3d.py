"""Native PyTorch ResNet18-style encoder for single-channel CT volumes."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


class BasicBlock3D(nn.Module):
    """Two-convolution residual block used by :class:`ResNet18Encoder3D`."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        identity = value
        output = self.relu(self.bn1(self.conv1(value)))
        output = self.bn2(self.conv2(output))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(output + identity)


class ResNet18Encoder3D(nn.Module):
    """ResNet18-class CT encoder with stage depths ``[2, 2, 2, 2]``.

    The backbone uses adaptive pooling and therefore does not assume a terminal
    spatial size.  Production ``roi_3d`` inputs are ``[B,1,64,160,160]``;
    smaller volumes are accepted for synthetic tests.
    """

    stage_depths = (2, 2, 2, 2)

    def __init__(
        self,
        embedding_dim: int = 128,
        pretrained: bool = False,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        base_width: int = 64,
    ) -> None:
        super().__init__()
        if pretrained:
            raise ValueError(
                "resnet18_3d has no bundled pretrained weights; use an explicit "
                "local pretraining checkpoint"
            )
        if (
            not isinstance(embedding_dim, int)
            or isinstance(embedding_dim, bool)
            or embedding_dim <= 0
        ):
            raise ValueError("embedding_dim must be a positive integer")
        if (
            not isinstance(base_width, int)
            or isinstance(base_width, bool)
            or base_width <= 0
        ):
            raise ValueError("base_width must be a positive integer")
        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool):
            raise ValueError("dropout must be a number in [0, 1]")
        if not 0.0 <= float(dropout) <= 1.0:
            raise ValueError("dropout must be in [0, 1]")

        self.embedding_dim = embedding_dim
        self.base_width = base_width
        self.input_channels = 1
        self.stem = nn.Sequential(
            nn.Conv3d(
                1,
                base_width,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm3d(base_width),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        self._current_channels = base_width
        self.layer1 = self._make_stage(base_width, blocks=2, stride=1)
        self.layer2 = self._make_stage(base_width * 2, blocks=2, stride=2)
        self.layer3 = self._make_stage(base_width * 4, blocks=2, stride=2)
        self.layer4 = self._make_stage(base_width * 8, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.backbone_feature_dim = base_width * 8
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.backbone_feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )
        self._backbone_frozen = False
        self._initialize_parameters()
        self.set_backbone_trainable(not freeze_backbone)

    def _make_stage(
        self, out_channels: int, *, blocks: int, stride: int
    ) -> nn.Sequential:
        layers = [BasicBlock3D(self._current_channels, out_channels, stride)]
        self._current_channels = out_channels
        layers.extend(BasicBlock3D(out_channels, out_channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def _initialize_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm3d, nn.LayerNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _backbone_modules(self) -> Iterable[nn.Module]:
        return (self.stem, self.layer1, self.layer2, self.layer3, self.layer4)

    @property
    def backbone_frozen(self) -> bool:
        return self._backbone_frozen

    def set_backbone_trainable(self, trainable: bool) -> None:
        if not isinstance(trainable, bool):
            raise ValueError("trainable must be bool")
        self._backbone_frozen = not trainable
        for module in self._backbone_modules():
            for parameter in module.parameters():
                parameter.requires_grad_(trainable)
            module.train(self.training and trainable)

    def set_transfer_policy(self, policy: str) -> None:
        """Apply a downstream ``frozen``, ``last_stage``, or ``full`` policy."""
        if policy not in {"frozen", "last_stage", "full"}:
            raise ValueError(
                "CT transfer policy must be 'frozen', 'last_stage', or 'full'"
            )
        self.set_backbone_trainable(policy == "full")
        if policy == "last_stage":
            for parameter in self.layer4.parameters():
                parameter.requires_grad_(True)
            self.layer4.train(self.training)
            self._backbone_frozen = False

    def train(self, mode: bool = True) -> ResNet18Encoder3D:
        super().train(mode)
        frozen_modules = [
            module
            for module in self._backbone_modules()
            if not any(parameter.requires_grad for parameter in module.parameters())
        ]
        for module in frozen_modules:
            module.eval()
        return self

    def forward(self, ct: torch.Tensor) -> torch.Tensor:
        single_sample = self._validate_input(ct)
        if single_sample:
            ct = ct.unsqueeze(0)
        value = self.stem(ct)
        value = self.layer1(value)
        value = self.layer2(value)
        value = self.layer3(value)
        value = self.layer4(value)
        embedding = self.projection(self.pool(value))
        if not torch.isfinite(embedding).all().item():
            raise ValueError("CT embedding contains non-finite values")
        return embedding.squeeze(0) if single_sample else embedding

    def _validate_input(self, ct: torch.Tensor) -> bool:
        if not isinstance(ct, torch.Tensor):
            raise ValueError(f"ct must be a torch.Tensor, got {type(ct).__name__}")
        if ct.ndim not in (4, 5):
            raise ValueError(
                "roi_3d ct must have shape [1,D,H,W] or [B,1,D,H,W], "
                f"got {tuple(ct.shape)}"
            )
        channel_axis = 0 if ct.ndim == 4 else 1
        if ct.shape[channel_axis] != self.input_channels:
            raise ValueError("roi_3d ct must have one channel")
        if min(ct.shape[-3:]) < 8:
            raise ValueError("roi_3d spatial dimensions must each be at least 8")
        if not torch.is_floating_point(ct):
            raise ValueError(f"roi_3d ct must be floating point, got {ct.dtype}")
        if not torch.isfinite(ct).all().item():
            raise ValueError("roi_3d ct must contain only finite values")
        minimum = float(ct.min().item())
        maximum = float(ct.max().item())
        if minimum < 0.0 or maximum > 1.0:
            raise ValueError(
                "roi_3d ct must follow the preprocessing range [0, 1], "
                f"got [{minimum}, {maximum}]"
            )
        parameter = next(self.parameters())
        if ct.device != parameter.device:
            raise ValueError(
                "ct and ResNet18Encoder3D must be on the same device, got "
                f"{ct.device} and {parameter.device}"
            )
        return ct.ndim == 4

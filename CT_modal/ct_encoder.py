#!/usr/bin/env python3
"""Encode an L3-centered 2.5D CT stack with a ResNet18 backbone."""

from __future__ import annotations

import argparse

import torch
from torch import nn

try:
    from torchvision.models import ResNet18_Weights, resnet18
except ImportError as error:  # pragma: no cover - depends on the runtime environment
    raise ImportError(
        "CTEncoder requires torchvision. Install a torchvision version compatible "
        "with the installed PyTorch version."
    ) from error


DEFAULT_INPUT_CHANNELS = 3
DEFAULT_EMBEDDING_DIM = 128
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CTEncoder(nn.Module):
    """Map a three-slice CT stack to a fixed-width embedding.

    The three channels are adjacent axial slices centered at L3, rather than
    RGB channels. Inputs must already follow the preprocessing contract used
    by ``CT_preparation/build_25d.py``: floating-point values in ``[0, 1]``
    with shape ``[3, H, W]`` or ``[B, 3, H, W]``.

    ImageNet normalization is applied inside this module immediately before
    the ResNet18 backbone. The backbone produces a 512-dimensional pooled
    feature, which is projected to ``embedding_dim`` for multimodal fusion.
    """

    def __init__(
        self,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        pretrained: bool = True,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        if not isinstance(embedding_dim, int) or isinstance(embedding_dim, bool):
            raise ValueError(
                f"embedding_dim must be a positive integer, got {embedding_dim!r}"
            )
        if embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be a positive integer, got {embedding_dim}"
            )
        if not isinstance(pretrained, bool):
            raise ValueError(f"pretrained must be bool, got {pretrained!r}")
        if not isinstance(freeze_backbone, bool):
            raise ValueError(
                f"freeze_backbone must be bool, got {freeze_backbone!r}"
            )
        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool):
            raise ValueError(f"dropout must be a number in [0, 1], got {dropout!r}")
        if not 0.0 <= float(dropout) <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")

        self.input_channels = DEFAULT_INPUT_CHANNELS
        self.embedding_dim = embedding_dim

        self.register_buffer(
            "image_mean",
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
        )

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone_feature_dim = int(self.backbone.fc.in_features)
        self.backbone.fc = nn.Identity()

        self.projection = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(self.backbone_feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

        self._backbone_frozen = False
        self.set_backbone_trainable(not freeze_backbone)

    @property
    def backbone_frozen(self) -> bool:
        """Return whether backbone gradients and BatchNorm updates are disabled."""
        return self._backbone_frozen

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Enable or disable backbone gradients without freezing the projection."""
        if not isinstance(trainable, bool):
            raise ValueError(f"trainable must be bool, got {trainable!r}")
        self._backbone_frozen = not trainable
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(trainable)
        if self._backbone_frozen:
            # Freezing parameters alone does not stop BatchNorm running-stat updates.
            self.backbone.eval()
        else:
            self.backbone.train(self.training)

    def train(self, mode: bool = True) -> CTEncoder:
        """Preserve an eval-mode backbone while a frozen projection is trained."""
        super().train(mode)
        if self._backbone_frozen:
            self.backbone.eval()
        return self

    def forward(self, ct_25d: torch.Tensor) -> torch.Tensor:
        """Return ``[embedding_dim]`` or ``[B, embedding_dim]`` CT features."""
        single_sample = self._validate_input(ct_25d)
        if single_sample:
            ct_25d = ct_25d.unsqueeze(0)

        mean = self.image_mean.to(dtype=ct_25d.dtype)
        std = self.image_std.to(dtype=ct_25d.dtype)
        normalized = (ct_25d - mean) / std
        embedding = self.projection(self.backbone(normalized))

        if not torch.isfinite(embedding).all().item():
            raise ValueError("CT embedding contains non-finite values")
        return embedding.squeeze(0) if single_sample else embedding

    def _validate_input(self, ct_25d: torch.Tensor) -> bool:
        if not isinstance(ct_25d, torch.Tensor):
            raise ValueError(
                f"ct_25d must be a torch.Tensor, got {type(ct_25d).__name__}"
            )
        if ct_25d.ndim not in (3, 4):
            raise ValueError(
                "ct_25d must have shape [3, H, W] or [B, 3, H, W], "
                f"got {tuple(ct_25d.shape)}"
            )
        channel_axis = 0 if ct_25d.ndim == 3 else 1
        if ct_25d.shape[channel_axis] != self.input_channels:
            raise ValueError(
                f"ct_25d must have {self.input_channels} channels, "
                f"got {ct_25d.shape[channel_axis]}"
            )
        if min(ct_25d.shape[-2:]) < 32:
            raise ValueError(
                "ct_25d spatial dimensions must both be at least 32, "
                f"got {tuple(ct_25d.shape[-2:])}"
            )
        if not torch.is_floating_point(ct_25d):
            raise ValueError(f"ct_25d must be floating point, got {ct_25d.dtype}")
        if not torch.isfinite(ct_25d).all().item():
            raise ValueError("ct_25d must contain only finite values")
        minimum = float(ct_25d.min().item())
        maximum = float(ct_25d.max().item())
        if minimum < 0.0 or maximum > 1.0:
            raise ValueError(
                "ct_25d must follow the preprocessing range [0, 1], "
                f"got [{minimum}, {maximum}]"
            )
        if ct_25d.device != self.image_mean.device:
            raise ValueError(
                "ct_25d and CTEncoder must be on the same device, got "
                f"{ct_25d.device} and {self.image_mean.device}"
            )
        return ct_25d.ndim == 3


def _run_smoke_test() -> None:
    """Exercise the encoder without downloading pretrained weights."""
    torch.manual_seed(42)
    model = CTEncoder(pretrained=False)
    model.eval()

    batch = torch.rand(2, 3, 224, 224, dtype=torch.float32)
    with torch.no_grad():
        batch_embedding = model(batch)
        single_embedding = model(batch[0])

    assert batch_embedding.shape == (2, DEFAULT_EMBEDDING_DIM)
    assert single_embedding.shape == (DEFAULT_EMBEDDING_DIM,)
    assert torch.isfinite(batch_embedding).all()
    assert torch.isfinite(single_embedding).all()

    frozen_model = CTEncoder(pretrained=False, freeze_backbone=True)
    frozen_model.train()
    assert frozen_model.backbone_frozen
    assert not frozen_model.backbone.training
    assert not any(parameter.requires_grad for parameter in frozen_model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in frozen_model.projection.parameters())

    invalid_channels = torch.rand(2, 5, 224, 224)
    try:
        model(invalid_channels)
    except ValueError as error:
        assert "3 channels" in str(error)
    else:
        raise AssertionError("five-channel input was not rejected")

    invalid_values = batch.clone()
    invalid_values[0, 0, 0, 0] = float("nan")
    try:
        model(invalid_values)
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("non-finite input was not rejected")

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"input batch shape:       {list(batch.shape)}")
    print(f"backbone feature dim:    {model.backbone_feature_dim}")
    print(f"batch embedding shape:   {list(batch_embedding.shape)}")
    print(f"single embedding shape:  {list(single_embedding.shape)}")
    print(f"total parameters:        {total_parameters}")
    print(f"trainable parameters:    {trainable_parameters}")
    print("frozen BatchNorm check:  passed")
    print("input validation checks: passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run local shape, freezing, and validation checks without weight download",
    )
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

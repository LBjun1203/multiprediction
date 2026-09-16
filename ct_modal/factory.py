"""Representation-aware CT encoder construction."""

from __future__ import annotations

from torch import nn

from ct_preparation.manifest import CTInputSpec, resolve_ct_input_spec

from .encoder_2d import CTEncoder2D
from .encoder_3d import CTEncoder3D


def build_ct_encoder(
    ct_config: CTInputSpec | dict[str, object] | None,
    *,
    embedding_dim: int,
    pretrained: bool,
    dropout: float,
    freeze_backbone: bool,
) -> nn.Module:
    spec = resolve_ct_input_spec(ct_config)
    common = {
        "embedding_dim": embedding_dim,
        "pretrained": pretrained,
        "dropout": dropout,
        "freeze_backbone": freeze_backbone,
    }
    if spec.encoder == "resnet18_2d":
        return CTEncoder2D(**common)
    if spec.encoder == "conv3d":
        return CTEncoder3D(**common)
    raise AssertionError(f"Validated but unsupported CT encoder: {spec.encoder}")

"""Joint CT-biochemistry contrastive model with task-specific projectors."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from biochemistry_modal.biochemistry_encoder import BiochemistryEncoder
from ct_modal.factory import build_ct_encoder
from ct_preparation.manifest import CTInputSpec, resolve_ct_input_spec
from multidataset.huashan_dataset import ALL_BIOMARKERS


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, output_dim) <= 0:
            raise ValueError("Projection dimensions must be positive")
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.layers(value).float(), dim=-1)


class CTBioContrastiveModel(nn.Module):
    """Learn shared CT-biochemistry embeddings from paired baseline samples."""

    def __init__(
        self,
        ct_config: CTInputSpec | Mapping[str, Any],
        *,
        ct_feature_dim: int = 128,
        ct_encoder_base_width: int = 64,
        ct_dropout: float = 0.1,
        bio_hidden_dim: int = 64,
        bio_feature_dim: int = 128,
        bio_dropout: float = 0.1,
        projection_hidden_dim: int = 128,
        projection_dim: int = 64,
        temperature: float = 0.07,
        learnable_temperature: bool = False,
        min_temperature: float = 0.01,
        max_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.ct_input_spec = resolve_ct_input_spec(ct_config)
        if self.ct_input_spec.representation != "roi_3d":
            raise ValueError("Contrastive pretraining currently requires roi_3d")
        if not 0 < min_temperature <= temperature <= max_temperature:
            raise ValueError(
                "temperature must be positive and within configured min/max bounds"
            )
        self.min_temperature = float(min_temperature)
        self.max_temperature = float(max_temperature)
        self.learnable_temperature = bool(learnable_temperature)
        self.ct_encoder = build_ct_encoder(
            self.ct_input_spec,
            embedding_dim=ct_feature_dim,
            pretrained=False,
            dropout=ct_dropout,
            freeze_backbone=False,
            base_width=ct_encoder_base_width,
        )
        self.bio_encoder = BiochemistryEncoder(
            input_dim=len(ALL_BIOMARKERS),
            hidden_dim=bio_hidden_dim,
            embedding_dim=bio_feature_dim,
            dropout=bio_dropout,
        )
        self.ct_projector = ProjectionHead(
            ct_feature_dim, projection_hidden_dim, projection_dim
        )
        self.bio_projector = ProjectionHead(
            bio_feature_dim, projection_hidden_dim, projection_dim
        )
        initial_log_temperature = torch.tensor(math.log(float(temperature)))
        if learnable_temperature:
            self.log_temperature = nn.Parameter(initial_log_temperature)
        else:
            self.register_buffer("log_temperature", initial_log_temperature)
        self.model_config = {
            "ct_config": self.ct_input_spec.to_dict(),
            "ct_feature_dim": ct_feature_dim,
            "ct_encoder_base_width": ct_encoder_base_width,
            "ct_dropout": ct_dropout,
            "bio_hidden_dim": bio_hidden_dim,
            "bio_feature_dim": bio_feature_dim,
            "bio_dropout": bio_dropout,
            "projection_hidden_dim": projection_hidden_dim,
            "projection_dim": projection_dim,
            "temperature": float(temperature),
            "learnable_temperature": bool(learnable_temperature),
            "min_temperature": self.min_temperature,
            "max_temperature": self.max_temperature,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "CTBioContrastiveModel":
        return cls(**dict(config))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.clamp(
            min=math.log(self.min_temperature), max=math.log(self.max_temperature)
        ).exp()

    def forward(
        self, ct: torch.Tensor, bio: torch.Tensor, bio_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        ct_feature = self.ct_encoder(ct)
        bio_feature = self.bio_encoder(bio, bio_mask)
        z_ct = self.ct_projector(ct_feature)
        z_bio = self.bio_projector(bio_feature)
        return {
            "ct_feature": ct_feature,
            "bio_feature": bio_feature,
            "z_ct": z_ct,
            "z_bio": z_bio,
            "temperature": self.temperature,
        }


class CTBiochemistryRegressionModel(nn.Module):
    """CT-only continuous regression control sharing the encoder contract."""

    def __init__(
        self,
        ct_config: CTInputSpec | Mapping[str, Any],
        *,
        ct_feature_dim: int = 128,
        ct_encoder_base_width: int = 64,
        ct_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ct_input_spec = resolve_ct_input_spec(ct_config)
        self.ct_encoder = build_ct_encoder(
            self.ct_input_spec,
            embedding_dim=ct_feature_dim,
            pretrained=False,
            dropout=ct_dropout,
            freeze_backbone=False,
            base_width=ct_encoder_base_width,
        )
        self.regression_head = nn.Linear(ct_feature_dim, len(ALL_BIOMARKERS))
        self.model_config = {
            "ct_config": self.ct_input_spec.to_dict(),
            "ct_feature_dim": ct_feature_dim,
            "ct_encoder_base_width": ct_encoder_base_width,
            "ct_dropout": ct_dropout,
        }

    def forward(self, ct: torch.Tensor) -> torch.Tensor:
        return self.regression_head(self.ct_encoder(ct))

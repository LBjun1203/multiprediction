#!/usr/bin/env python3
"""CT-free neural comparator for the Huashan forecasting pipeline."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from biochemistry_modal.biochemistry_encoder import BiochemistryEncoder
from multidataset.huashan_dataset import ALL_BIOMARKERS


class TabularForecastModel(nn.Module):
    """Predict from baseline biochemistry/mask, age, sex, and query time."""

    def __init__(
        self,
        bio_hidden_dim: int = 64,
        bio_embedding_dim: int = 128,
        demo_hidden_dim: int = 16,
        demo_embedding_dim: int = 32,
        fusion_hidden_dim: int = 128,
        fusion_dim: int = 128,
        time_hidden_dim: int = 32,
        time_embedding_dim: int = 32,
        decoder_hidden_dim: int = 128,
        decoder_bottleneck_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dimensions = {
            "bio_hidden_dim": bio_hidden_dim,
            "bio_embedding_dim": bio_embedding_dim,
            "demo_hidden_dim": demo_hidden_dim,
            "demo_embedding_dim": demo_embedding_dim,
            "fusion_hidden_dim": fusion_hidden_dim,
            "fusion_dim": fusion_dim,
            "time_hidden_dim": time_hidden_dim,
            "time_embedding_dim": time_embedding_dim,
            "decoder_hidden_dim": decoder_hidden_dim,
            "decoder_bottleneck_dim": decoder_bottleneck_dim,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool):
            raise TypeError(f"dropout must be a number in [0, 1], got {dropout!r}")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")

        self.bio_encoder = BiochemistryEncoder(
            input_dim=len(ALL_BIOMARKERS),
            hidden_dim=bio_hidden_dim,
            embedding_dim=bio_embedding_dim,
            dropout=dropout,
        )
        self.demographics_encoder = nn.Sequential(
            nn.Linear(2, demo_hidden_dim),
            nn.GELU(),
            nn.Linear(demo_hidden_dim, demo_embedding_dim),
            nn.GELU(),
        )
        self.baseline_fusion = nn.Sequential(
            nn.Linear(bio_embedding_dim + demo_embedding_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_dim),
            nn.GELU(),
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(1, time_hidden_dim),
            nn.GELU(),
            nn.Linear(time_hidden_dim, time_embedding_dim),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(fusion_dim + time_embedding_dim, decoder_hidden_dim),
            nn.LayerNorm(decoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, decoder_bottleneck_dim),
            nn.GELU(),
            nn.Linear(decoder_bottleneck_dim, len(ALL_BIOMARKERS)),
        )
        self.model_config = {
            **dimensions,
            "dropout": float(dropout),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> TabularForecastModel:
        return cls(**dict(config))

    @staticmethod
    def _validate_tensor(
        name: str,
        tensor: torch.Tensor,
        last_dimension: int,
        reference: torch.Tensor | None = None,
    ) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim not in (1, 2) or tensor.shape[-1] != last_dimension:
            raise ValueError(
                f"{name} must have shape [{last_dimension}] or "
                f"[batch, {last_dimension}], got {tuple(tensor.shape)}"
            )
        if not torch.is_floating_point(tensor) or not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain finite floating-point values")
        if reference is not None:
            if tensor.shape[:-1] != reference.shape[:-1]:
                raise ValueError("All tabular model inputs must share batch dimensions")
            if tensor.device != reference.device or tensor.dtype != reference.dtype:
                raise ValueError("All tabular model inputs must share device and dtype")

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        return_features: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        required = {"bio_base", "base_mask", "demographics", "time"}
        missing = sorted(required - set(batch))
        if missing:
            raise ValueError(f"Model batch is missing keys: {missing}")
        bio = batch["bio_base"]
        mask = batch["base_mask"]
        demo = batch["demographics"]
        time = batch["time"]
        self._validate_tensor("bio_base", bio, len(ALL_BIOMARKERS))
        self._validate_tensor("base_mask", mask, len(ALL_BIOMARKERS), bio)
        self._validate_tensor("demographics", demo, 2, bio)
        self._validate_tensor("time", time, 1, bio)
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("base_mask must contain only binary values 0 or 1")

        bio_embedding = self.bio_encoder(bio, mask)
        demo_embedding = self.demographics_encoder(demo)
        baseline_state = self.baseline_fusion(
            torch.cat([bio_embedding, demo_embedding], dim=-1)
        )
        time_embedding = self.time_encoder(time)
        prediction = self.decoder(torch.cat([baseline_state, time_embedding], dim=-1))
        if not torch.isfinite(prediction).all():
            raise ValueError("Model prediction contains non-finite values")
        if return_features:
            return {
                "prediction": prediction,
                "bio_embedding": bio_embedding,
                "demo_embedding": demo_embedding,
                "baseline_state": baseline_state,
                "time_embedding": time_embedding,
            }
        return prediction

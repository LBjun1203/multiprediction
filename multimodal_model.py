#!/usr/bin/env python3
"""End-to-end Huashan multimodal forecasting model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from biochemistry_modal.biochemistry_encoder import (
    BIOMARKERS as BIOCHEMISTRY_BIOMARKERS,
)
from biochemistry_modal.biochemistry_encoder import BiochemistryEncoder
from ct_modal.factory import build_ct_encoder
from ct_preparation.manifest import CTInputSpec, resolve_ct_input_spec
from fusion_query_modal.fusion_time_query import BIOMARKERS as FUSION_BIOMARKERS
from fusion_query_modal.fusion_time_query import FusionTimeQueryHead
from multidataset.huashan_dataset import ALL_BIOMARKERS


class HuashanForecastModel(nn.Module):
    """Compose CT, biochemistry, demographics, time, and prediction modules."""

    def __init__(
        self,
        ct_embedding_dim: int = 128,
        bio_embedding_dim: int = 128,
        pretrained_ct: bool = True,
        freeze_ct_backbone: bool = True,
        ct_dropout: float = 0.1,
        bio_hidden_dim: int = 64,
        bio_dropout: float = 0.1,
        fusion_dropout: float = 0.1,
        ct_encoder_base_width: int = 64,
        ct_config: CTInputSpec | Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._validate_contract()
        self.ct_input_spec = resolve_ct_input_spec(ct_config)
        self._ct_encoder_fully_frozen = False
        self.ct_encoder = build_ct_encoder(
            self.ct_input_spec,
            embedding_dim=ct_embedding_dim,
            pretrained=pretrained_ct,
            dropout=ct_dropout,
            freeze_backbone=freeze_ct_backbone,
            base_width=ct_encoder_base_width,
        )
        self.bio_encoder = BiochemistryEncoder(
            input_dim=len(ALL_BIOMARKERS),
            hidden_dim=bio_hidden_dim,
            embedding_dim=bio_embedding_dim,
            dropout=bio_dropout,
        )
        self.fusion_head = FusionTimeQueryHead(
            ct_embedding_dim=ct_embedding_dim,
            bio_embedding_dim=bio_embedding_dim,
            num_outputs=len(ALL_BIOMARKERS),
            dropout=fusion_dropout,
        )
        self.model_config = {
            "ct_embedding_dim": ct_embedding_dim,
            "bio_embedding_dim": bio_embedding_dim,
            "pretrained_ct": pretrained_ct,
            "freeze_ct_backbone": freeze_ct_backbone,
            "ct_dropout": ct_dropout,
            "bio_hidden_dim": bio_hidden_dim,
            "bio_dropout": bio_dropout,
            "fusion_dropout": fusion_dropout,
            "ct_encoder_base_width": ct_encoder_base_width,
            "ct_config": self.ct_input_spec.to_dict(),
        }

    @staticmethod
    def _validate_contract() -> None:
        expected = list(ALL_BIOMARKERS)
        if list(BIOCHEMISTRY_BIOMARKERS) != expected:
            raise RuntimeError(
                "BiochemistryEncoder biomarker order differs from the Dataset contract"
            )
        if list(FUSION_BIOMARKERS) != expected:
            raise RuntimeError(
                "FusionTimeQueryHead biomarker order differs from the Dataset contract"
            )

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        pretrained_override: bool | None = None,
    ) -> HuashanForecastModel:
        values = dict(config)
        if pretrained_override is not None:
            values["pretrained_ct"] = pretrained_override
        return cls(**values)

    def set_ct_backbone_trainable(self, trainable: bool) -> None:
        self.ct_encoder.set_backbone_trainable(trainable)
        self.model_config["freeze_ct_backbone"] = not trainable

    def set_ct_transfer_policy(self, policy: str) -> None:
        if not hasattr(self.ct_encoder, "set_transfer_policy"):
            if policy != "frozen":
                raise ValueError(
                    f"CT encoder {self.ct_input_spec.encoder!r} does not support "
                    f"transfer policy {policy!r}"
                )
            for parameter in self.ct_encoder.parameters():
                parameter.requires_grad_(False)
            self.ct_encoder.eval()
        else:
            self.ct_encoder.set_transfer_policy(policy)
        self._ct_encoder_fully_frozen = policy == "frozen"
        self.model_config["ct_transfer_policy"] = policy

    def train(self, mode: bool = True) -> "HuashanForecastModel":
        super().train(mode)
        if self._ct_encoder_fully_frozen:
            self.ct_encoder.eval()
        return self

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        return_features: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        required = {"ct", "bio_base", "base_mask", "demographics", "time"}
        missing = sorted(required - set(batch))
        if missing:
            raise ValueError(f"Model batch is missing keys: {missing}")

        ct_embedding = self.ct_encoder(batch["ct"])
        bio_embedding = self.bio_encoder(batch["bio_base"], batch["base_mask"])
        result = self.fusion_head(
            ct_embedding,
            bio_embedding,
            batch["demographics"],
            batch["time"],
            return_features=return_features,
        )
        prediction = result["prediction"] if isinstance(result, dict) else result
        if prediction.shape[-1] != len(ALL_BIOMARKERS):
            raise RuntimeError(
                f"Model produced {prediction.shape[-1]} outputs; expected "
                f"{len(ALL_BIOMARKERS)}"
            )
        if not torch.isfinite(prediction).all().item():
            raise ValueError("Model prediction contains non-finite values")
        if isinstance(result, dict):
            return {
                **result,
                "ct_embedding": ct_embedding,
                "bio_embedding": bio_embedding,
            }
        return prediction

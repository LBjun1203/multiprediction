"""Atomic, validated pretraining checkpoint serialization."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from multidataset.huashan_dataset import ALL_BIOMARKERS
from pretraining.contrastive_model import CTBioContrastiveModel

CHECKPOINT_TYPE = "ct_bio_contrastive_pretraining"
CHECKPOINT_VERSION = 1
REQUIRED_KEYS = {
    "checkpoint_type",
    "checkpoint_version",
    "ct_encoder_state",
    "bio_encoder_state",
    "ct_projector_state",
    "bio_projector_state",
    "temperature_state",
    "model_config",
    "ct_config",
    "biomarker_order",
    "normalization_stats",
    "pretraining_split_provenance",
    "optimizer_state",
    "scaler_state",
    "epoch",
    "best_validation_metric",
    "history",
    "rng_state",
}


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def build_checkpoint(
    model: CTBioContrastiveModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    epoch: int,
    best_validation_metric: float,
    normalization_stats: dict[str, Any],
    split_provenance: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "checkpoint_type": CHECKPOINT_TYPE,
        "checkpoint_version": CHECKPOINT_VERSION,
        "ct_encoder_state": model.ct_encoder.state_dict(),
        "bio_encoder_state": model.bio_encoder.state_dict(),
        "ct_projector_state": model.ct_projector.state_dict(),
        "bio_projector_state": model.bio_projector.state_dict(),
        "temperature_state": {"log_temperature": model.log_temperature.detach()},
        "model_config": dict(model.model_config),
        "ct_config": model.ct_input_spec.to_dict(),
        "biomarker_order": list(ALL_BIOMARKERS),
        "normalization_stats": normalization_stats,
        "pretraining_split_provenance": split_provenance,
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": int(epoch),
        "best_validation_metric": float(best_validation_metric),
        "history": list(history),
        "rng_state": capture_rng_state(),
    }


def validate_checkpoint(checkpoint: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_KEYS - set(checkpoint))
    if missing:
        raise ValueError(f"Pretraining checkpoint is missing keys: {missing}")
    if checkpoint["checkpoint_type"] != CHECKPOINT_TYPE:
        raise ValueError(
            f"Expected checkpoint_type={CHECKPOINT_TYPE!r}, got "
            f"{checkpoint['checkpoint_type']!r}"
        )
    if checkpoint["checkpoint_version"] != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported pretraining checkpoint version: {checkpoint['checkpoint_version']}"
        )
    if checkpoint["biomarker_order"] != list(ALL_BIOMARKERS):
        raise ValueError("Pretraining checkpoint biomarker order is incompatible")


def save_checkpoint(path: str | Path, checkpoint: dict[str, Any]) -> None:
    validate_checkpoint(checkpoint)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(output)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    restore_rng: bool = False,
) -> tuple[CTBioContrastiveModel, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Pretraining checkpoint must contain a mapping")
    validate_checkpoint(checkpoint)
    model = CTBioContrastiveModel.from_config(checkpoint["model_config"])
    model.ct_encoder.load_state_dict(checkpoint["ct_encoder_state"], strict=True)
    model.bio_encoder.load_state_dict(checkpoint["bio_encoder_state"], strict=True)
    model.ct_projector.load_state_dict(checkpoint["ct_projector_state"], strict=True)
    model.bio_projector.load_state_dict(checkpoint["bio_projector_state"], strict=True)
    with torch.no_grad():
        model.log_temperature.copy_(
            checkpoint["temperature_state"]["log_temperature"]
        )
    model.to(map_location)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])
    if restore_rng:
        restore_rng_state(checkpoint["rng_state"])
    return model, checkpoint

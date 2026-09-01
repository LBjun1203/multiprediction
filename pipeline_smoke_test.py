#!/usr/bin/env python3
"""Run a synthetic Dataset-to-checkpoint integration smoke test."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from evaluate import load_checkpoint_model
from fusion_query_modal.fusion_time_query import masked_huber_loss
from multidataset.huashan_dataset import (
    ALL_BIOMARKERS,
    CORE_BIOMARKERS,
    HuashanMultimodalDataset,
    fit_normalization_stats,
    make_patient_balanced_sampler,
)
from multimodal_model import HuashanForecastModel


def synthetic_row(
    patient_id: str,
    ct_path: Path,
    offset: float,
    t_days: int,
    base_cysc_observed: bool,
    target_cysc_observed: bool,
) -> dict[str, object]:
    row: dict[str, object] = {
        "patient_id": patient_id,
        "ct_path": "synthetic://ct",
        "npy_25d_path": str(ct_path),
        "age": 55.0 + offset,
        "sex_code": int(offset) % 2,
        "t_days": t_days,
        "base_CysC": 1.0 + 0.01 * offset if base_cysc_observed else np.nan,
        "base_CysC_mask": int(base_cysc_observed),
        "target_CysC": 1.1 + 0.01 * offset if target_cysc_observed else np.nan,
        "target_CysC_mask": int(target_cysc_observed),
    }
    for index, name in enumerate(CORE_BIOMARKERS):
        row[f"base_{name}"] = 1.0 + index + 0.1 * offset
        row[f"target_{name}"] = 1.2 + index + 0.1 * offset + t_days / 10000.0
    return row


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    with tempfile.TemporaryDirectory(prefix="huashan_pipeline_smoke_") as temp_name:
        temp = Path(temp_name)
        paths: dict[str, Path] = {}
        for index, patient_id in enumerate(("p1", "p2", "p3", "p4")):
            path = temp / f"{patient_id}.npy"
            array = np.random.default_rng(index).random((3, 64, 64), dtype=np.float32)
            np.save(path, array, allow_pickle=False)
            paths[patient_id] = path

        rows = [
            synthetic_row("p1", paths["p1"], 0, 90, True, True),
            synthetic_row("p1", paths["p1"], 0, 365, True, False),
            synthetic_row("p2", paths["p2"], 2, 120, False, True),
            synthetic_row("p2", paths["p2"], 2, 500, False, True),
            synthetic_row("p3", paths["p3"], 4, 180, True, True),
            synthetic_row("p4", paths["p4"], 6, 240, True, False),
        ]
        dataframe = pd.DataFrame(rows)
        train_frame = dataframe[dataframe["patient_id"].isin({"p1", "p2"})]
        val_frame = dataframe[dataframe["patient_id"] == "p3"]
        test_frame = dataframe[dataframe["patient_id"] == "p4"]
        id_sets = [
            set(frame["patient_id"]) for frame in (train_frame, val_frame, test_frame)
        ]
        assert not id_sets[0] & id_sets[1]
        assert not id_sets[0] & id_sets[2]
        assert not id_sets[1] & id_sets[2]

        stats = fit_normalization_stats(train_frame)
        datasets = [
            HuashanMultimodalDataset(frame, stats, expected_ct_shape=(3, 64, 64))
            for frame in (train_frame, val_frame, test_frame)
        ]
        sampler = make_patient_balanced_sampler(datasets[0])
        loader = DataLoader(datasets[0], batch_size=2, sampler=sampler)
        batch = next(iter(loader))

        model = HuashanForecastModel(pretrained_ct=False, freeze_ct_backbone=True)
        model.train()
        optimizer = AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=1e-3,
        )
        prediction = model(batch)
        assert prediction.shape == (2, len(ALL_BIOMARKERS))
        loss = masked_huber_loss(prediction, batch["target"], batch["target_mask"])
        assert loss.ndim == 0 and torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        finite_gradients = [
            torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        assert finite_gradients and all(finite_gradients)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            reference = model(batch)
        checkpoint_path = temp / "smoke.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": 1,
                "best_val_loss": float(loss.item()),
                "model_config": model.model_config,
                "biomarker_order": ALL_BIOMARKERS,
                "normalization_stats": stats,
            },
            checkpoint_path,
        )
        restored, _ = load_checkpoint_model(checkpoint_path, torch.device("cpu"))
        with torch.no_grad():
            restored_prediction = restored(batch)
        assert torch.allclose(reference, restored_prediction, rtol=0.0, atol=1e-6)

    print("Pipeline smoke test passed:")
    print("  Dataset -> DataLoader -> CT/Bio encoders -> fusion -> masked loss")
    print("  backward -> optimizer step -> checkpoint save/load")
    print("  patient-level split isolation and deterministic reload")


if __name__ == "__main__":
    main()

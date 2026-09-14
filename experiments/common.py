"""Shared data and metric utilities for add-only comparison experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from multidataset.huashan_dataset import ALL_BIOMARKERS, CORE_BIOMARKERS
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TabularArrays:
    """In-memory tabular view of one CT-eligible dataset split."""

    patient_ids: np.ndarray
    time_years: np.ndarray
    bio_base: np.ndarray
    base_mask: np.ndarray
    demographics: np.ndarray
    features: np.ndarray
    target: np.ndarray
    target_mask: np.ndarray
    dataframe: pd.DataFrame


def atomic_write_json(payload: Any, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def atomic_write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(output)


def _standardize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (values.astype(np.float64) - float(mean)) / float(std)


def tabular_arrays_from_dataset(dataset: Any) -> TabularArrays:
    """Build the exact non-CT tensors used by the neural Dataset without CT I/O."""
    frame = dataset.dataframe.reset_index(drop=True).copy()
    stats = dataset.normalization_stats
    count = len(frame)

    bio_base = np.empty((count, len(ALL_BIOMARKERS)), dtype=np.float32)
    base_mask = np.ones_like(bio_base)
    target = np.empty_like(bio_base)
    target_mask = np.ones_like(bio_base)

    for marker_index, marker in enumerate(CORE_BIOMARKERS):
        base_entry = stats["base"][marker]
        target_entry = stats["target"][marker]
        bio_base[:, marker_index] = _standardize(
            frame[f"base_{marker}"].to_numpy(float),
            base_entry["mean"],
            base_entry["std"],
        )
        target[:, marker_index] = _standardize(
            frame[f"target_{marker}"].to_numpy(float),
            target_entry["mean"],
            target_entry["std"],
        )

    cysc_index = len(ALL_BIOMARKERS) - 1
    base_cysc_stats = stats["base"]["CysC"]
    base_cysc_observed = frame["base_CysC_mask"].to_numpy(float)
    base_cysc = frame["base_CysC"].to_numpy(float)
    base_cysc = np.where(
        base_cysc_observed == 1,
        base_cysc,
        float(base_cysc_stats["median"]),
    )
    bio_base[:, cysc_index] = _standardize(
        base_cysc, base_cysc_stats["mean"], base_cysc_stats["std"]
    )
    base_mask[:, cysc_index] = base_cysc_observed

    target_cysc_observed = frame["target_CysC_mask"].to_numpy(float)
    target_cysc_stats = stats["target"]["CysC"]
    target_cysc = frame["target_CysC"].to_numpy(float)
    target[:, cysc_index] = np.where(
        target_cysc_observed == 1,
        _standardize(
            np.nan_to_num(target_cysc, nan=float(target_cysc_stats["mean"])),
            target_cysc_stats["mean"],
            target_cysc_stats["std"],
        ),
        0.0,
    )
    target_mask[:, cysc_index] = target_cysc_observed

    age_stats = stats["age"]
    demographics = np.column_stack(
        [
            _standardize(frame["age"].to_numpy(float), age_stats["mean"], age_stats["std"]),
            frame["sex_code"].to_numpy(float),
        ]
    ).astype(np.float32)
    time_years = (frame["t_days"].to_numpy(float) / 365.25).astype(np.float32)
    features = np.column_stack(
        [bio_base, base_mask, demographics, time_years[:, None]]
    ).astype(np.float32)

    arrays = (bio_base, base_mask, demographics, features, target, target_mask)
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("Tabular arrays contain non-finite values")
    if not np.all((base_mask == 0) | (base_mask == 1)):
        raise ValueError("base_mask must be binary")
    if not np.all((target_mask == 0) | (target_mask == 1)):
        raise ValueError("target_mask must be binary")

    return TabularArrays(
        patient_ids=frame["patient_id"].astype(str).to_numpy(),
        time_years=time_years,
        bio_base=bio_base,
        base_mask=base_mask,
        demographics=demographics,
        features=features,
        target=target,
        target_mask=target_mask,
        dataframe=frame,
    )


class TabularDatasetView(Dataset):
    """PyTorch Dataset that reuses the final CT-eligible cohort without reading CT."""

    def __init__(self, source_dataset: Any) -> None:
        self.arrays = tabular_arrays_from_dataset(source_dataset)
        self.normalization_stats = source_dataset.normalization_stats
        self.patient_ids = self.arrays.patient_ids.astype(str).tolist()
        self.dataframe = self.arrays.dataframe

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int) -> dict[str, str | torch.Tensor]:
        return {
            "patient_id": self.patient_ids[index],
            "bio_base": torch.from_numpy(self.arrays.bio_base[index]),
            "base_mask": torch.from_numpy(self.arrays.base_mask[index]),
            "demographics": torch.from_numpy(self.arrays.demographics[index]),
            "time": torch.tensor([self.arrays.time_years[index]], dtype=torch.float32),
            "target": torch.from_numpy(self.arrays.target[index]),
            "target_mask": torch.from_numpy(self.arrays.target_mask[index]),
        }


def patient_balanced_weights(patient_ids: np.ndarray) -> np.ndarray:
    ids, counts = np.unique(patient_ids.astype(str), return_counts=True)
    count_map = dict(zip(ids.tolist(), counts.tolist(), strict=True))
    return np.asarray([1.0 / count_map[str(value)] for value in patient_ids], dtype=np.float64)


def masked_huber_numpy(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    delta: float = 1.0,
) -> float:
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("prediction, target, and mask must have identical shapes")
    valid = mask.astype(bool)
    if not valid.any():
        raise ValueError("mask contains no valid entries")
    error = np.abs(prediction.astype(np.float64) - target.astype(np.float64))
    loss = np.where(error <= delta, 0.5 * error**2, delta * (error - 0.5 * delta))
    return float(loss[valid].mean())


def _pearson_and_r2(prediction: np.ndarray, target: np.ndarray) -> tuple[float | None, float | None]:
    if prediction.size <= 1 or np.std(prediction) <= 0 or np.std(target) <= 0:
        pearson = None
    else:
        pearson = float(np.corrcoef(prediction, target)[0, 1])
    denominator = float(np.sum((target - target.mean()) ** 2))
    r_squared = (
        float(1.0 - np.sum((prediction - target) ** 2) / denominator)
        if denominator > 0
        else None
    )
    return pearson, r_squared


def evaluate_prediction_arrays(
    prediction: np.ndarray,
    arrays: TabularArrays,
    normalization_stats: dict[str, Any],
    evaluation_mask: np.ndarray | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate standardized predictions and return metrics plus clinical CSV rows."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = arrays.target.astype(np.float64)
    mask = arrays.target_mask.astype(bool)
    if evaluation_mask is not None:
        if evaluation_mask.shape != mask.shape:
            raise ValueError("evaluation_mask shape differs from target_mask")
        mask &= evaluation_mask.astype(bool)
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction has shape {prediction.shape}, expected {target.shape}"
        )
    if not np.isfinite(prediction).all():
        raise ValueError("prediction contains non-finite values")

    clinical_prediction = np.empty_like(prediction)
    clinical_target = np.empty_like(target)
    per_biomarker: dict[str, dict[str, float | int | None]] = {}
    rows: dict[str, Any] = {
        "patient_id": arrays.patient_ids.astype(str),
        "time_years": arrays.time_years.astype(float),
    }

    for marker_index, marker in enumerate(ALL_BIOMARKERS):
        entry = normalization_stats["target"][marker]
        clinical_prediction[:, marker_index] = (
            prediction[:, marker_index] * float(entry["std"]) + float(entry["mean"])
        )
        clinical_target[:, marker_index] = (
            target[:, marker_index] * float(entry["std"]) + float(entry["mean"])
        )
        valid = mask[:, marker_index]
        if not valid.any():
            per_biomarker[marker] = {
                "count": 0,
                "normalized_mae": None,
                "normalized_rmse": None,
                "clinical_mae": None,
                "clinical_rmse": None,
                "clinical_bias": None,
                "pearson_r": None,
                "r_squared": None,
            }
        else:
            normalized_error = prediction[valid, marker_index] - target[valid, marker_index]
            clinical_error = (
                clinical_prediction[valid, marker_index]
                - clinical_target[valid, marker_index]
            )
            pearson, r_squared = _pearson_and_r2(
                clinical_prediction[valid, marker_index],
                clinical_target[valid, marker_index],
            )
            per_biomarker[marker] = {
                "count": int(valid.sum()),
                "normalized_mae": float(np.mean(np.abs(normalized_error))),
                "normalized_rmse": float(np.sqrt(np.mean(normalized_error**2))),
                "clinical_mae": float(np.mean(np.abs(clinical_error))),
                "clinical_rmse": float(np.sqrt(np.mean(clinical_error**2))),
                "clinical_bias": float(np.mean(clinical_error)),
                "pearson_r": pearson,
                "r_squared": r_squared,
            }
        rows[f"{marker}_mask"] = valid.astype(int)
        rows[f"{marker}_prediction"] = clinical_prediction[:, marker_index]
        rows[f"{marker}_target"] = np.where(
            valid, clinical_target[:, marker_index], np.nan
        )

    metric_names = (
        "normalized_mae",
        "normalized_rmse",
        "clinical_mae",
        "clinical_rmse",
    )
    macro_average = {
        metric: float(
            np.mean(
                [
                    values[metric]
                    for values in per_biomarker.values()
                    if values[metric] is not None
                ]
            )
        )
        for metric in metric_names
    }
    positive_r_squared = sum(
        values["r_squared"] is not None and values["r_squared"] > 0
        for values in per_biomarker.values()
    )
    metrics = {
        "masked_huber_loss": masked_huber_numpy(prediction, target, mask),
        "observations": len(arrays.patient_ids),
        "patients": len(np.unique(arrays.patient_ids.astype(str))),
        "supervised_elements": int(mask.sum()),
        "positive_r_squared": int(positive_r_squared),
        "macro_average": macro_average,
        "per_biomarker": per_biomarker,
    }
    return metrics, pd.DataFrame(rows)


def write_evaluation(
    output_dir: str | Path,
    metrics: dict[str, Any],
    prediction_frame: pd.DataFrame,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(metrics, output / "test_metrics.json")
    atomic_write_csv(prediction_frame, output / "test_predictions.csv")


def enrich_metrics_from_prediction_csv(
    metrics: dict[str, Any], prediction_frame: pd.DataFrame
) -> dict[str, Any]:
    """Add correlation/calibration fields to existing pipeline metrics."""
    enriched = json.loads(json.dumps(metrics))
    positive = 0
    for marker, values in enriched["per_biomarker"].items():
        valid = prediction_frame[f"{marker}_mask"].eq(1) & prediction_frame[
            f"{marker}_target"
        ].notna()
        prediction = prediction_frame.loc[valid, f"{marker}_prediction"].to_numpy(float)
        target = prediction_frame.loc[valid, f"{marker}_target"].to_numpy(float)
        if prediction.size:
            pearson, r_squared = _pearson_and_r2(prediction, target)
            values["clinical_bias"] = float(np.mean(prediction - target))
            values["pearson_r"] = pearson
            values["r_squared"] = r_squared
            positive += r_squared is not None and r_squared > 0
        else:
            values["clinical_bias"] = None
            values["pearson_r"] = None
            values["r_squared"] = None
    enriched["positive_r_squared"] = int(positive)
    enriched.setdefault("patients", int(prediction_frame["patient_id"].nunique()))
    return enriched

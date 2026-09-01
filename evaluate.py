#!/usr/bin/env python3
"""Evaluate a Huashan checkpoint and export mask-aware predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fusion_query_modal.fusion_time_query import masked_huber_loss
from multidataset.huashan_dataset import ALL_BIOMARKERS, build_datasets
from multimodal_model import HuashanForecastModel

TENSOR_BATCH_KEYS = (
    "ct_25d",
    "bio_base",
    "base_mask",
    "demographics",
    "time",
    "target",
    "target_mask",
)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in TENSOR_BATCH_KEYS:
        moved[key] = batch[key].to(device, non_blocking=device.type == "cuda")
    return moved


def load_checkpoint_model(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[HuashanForecastModel, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    required = {"model_state", "model_config", "biomarker_order"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint is missing keys: {missing}")
    if checkpoint["biomarker_order"] != ALL_BIOMARKERS:
        raise ValueError("Checkpoint biomarker order does not match this project")
    model = HuashanForecastModel.from_config(
        checkpoint["model_config"], pretrained_override=False
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def evaluate_loader(
    model: HuashanForecastModel,
    loader: DataLoader,
    normalization_stats: dict[str, Any],
    device: torch.device,
    prediction_csv: str | Path | None = None,
) -> dict[str, Any]:
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    patient_ids: list[str] = []
    time_years: list[float] = []
    weighted_loss_sum = 0.0
    supervised_count = 0.0

    model.eval()
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch_to_device(raw_batch, device)
            prediction = model(batch)
            valid_count = float(batch["target_mask"].sum().item())
            loss = masked_huber_loss(prediction, batch["target"], batch["target_mask"])
            weighted_loss_sum += float(loss.item()) * valid_count
            supervised_count += valid_count
            predictions.append(prediction.cpu())
            targets.append(batch["target"].cpu())
            masks.append(batch["target_mask"].cpu())
            patient_ids.extend(str(value) for value in raw_batch["patient_id"])
            time_years.extend(float(value) for value in raw_batch["time"].view(-1))

    if not predictions or supervised_count <= 0:
        raise ValueError("Evaluation loader produced no supervised observations")
    prediction = torch.cat(predictions).numpy()
    target = torch.cat(targets).numpy()
    mask = torch.cat(masks).numpy().astype(bool)

    per_biomarker: dict[str, dict[str, float | int | None]] = {}
    prediction_rows: list[dict[str, Any]] = []
    clinical_predictions = np.empty_like(prediction, dtype=np.float64)
    clinical_targets = np.empty_like(target, dtype=np.float64)
    for index, name in enumerate(ALL_BIOMARKERS):
        stats = normalization_stats["target"][name]
        clinical_predictions[:, index] = (
            prediction[:, index] * stats["std"] + stats["mean"]
        )
        clinical_targets[:, index] = target[:, index] * stats["std"] + stats["mean"]
        valid = mask[:, index]
        if not valid.any():
            per_biomarker[name] = {
                "count": 0,
                "normalized_mae": None,
                "normalized_rmse": None,
                "clinical_mae": None,
                "clinical_rmse": None,
            }
            continue
        normalized_error = prediction[valid, index] - target[valid, index]
        clinical_error = (
            clinical_predictions[valid, index] - clinical_targets[valid, index]
        )
        per_biomarker[name] = {
            "count": int(valid.sum()),
            "normalized_mae": float(np.mean(np.abs(normalized_error))),
            "normalized_rmse": float(np.sqrt(np.mean(normalized_error**2))),
            "clinical_mae": float(np.mean(np.abs(clinical_error))),
            "clinical_rmse": float(np.sqrt(np.mean(clinical_error**2))),
        }

    for row_index, patient_id in enumerate(patient_ids):
        row: dict[str, Any] = {
            "patient_id": patient_id,
            "time_years": time_years[row_index],
        }
        for marker_index, name in enumerate(ALL_BIOMARKERS):
            observed = bool(mask[row_index, marker_index])
            row[f"{name}_mask"] = int(observed)
            row[f"{name}_prediction"] = clinical_predictions[row_index, marker_index]
            row[f"{name}_target"] = (
                clinical_targets[row_index, marker_index] if observed else np.nan
            )
        prediction_rows.append(row)

    if prediction_csv is not None:
        output = Path(prediction_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(prediction_rows).to_csv(output, index=False)

    metric_names = (
        "normalized_mae",
        "normalized_rmse",
        "clinical_mae",
        "clinical_rmse",
    )
    macro = {
        metric: float(
            np.mean(
                [
                    entry[metric]
                    for entry in per_biomarker.values()
                    if entry[metric] is not None
                ]
            )
        )
        for metric in metric_names
    }
    return {
        "masked_huber_loss": weighted_loss_sum / supervised_count,
        "observations": len(patient_ids),
        "macro_average": macro,
        "per_biomarker": per_biomarker,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--ct-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--split-file", type=Path, default=Path("multidataset/patient_split.csv")
    )
    parser.add_argument(
        "--stats-file", type=Path, default=Path("multidataset/normalization.json")
    )
    parser.add_argument("--sheet", default="model_long_0_2y")
    parser.add_argument("--ct-root", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--predictions", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--metrics", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("--batch-size must be positive and --num-workers non-negative")
    if args.image_size < 32:
        raise ValueError("--image-size must be at least 32")
    device = resolve_device(args.device)
    datasets = build_datasets(
        xlsx_path=args.input,
        ct_manifest_path=args.ct_manifest,
        split_csv=args.split_file,
        stats_path=args.stats_file,
        sheet_name=args.sheet,
        fit_stats=False,
        expected_ct_shape=(3, args.image_size, args.image_size),
        ct_root=args.ct_root,
    )
    dataset = datasets[{"val": 1, "test": 2}[args.split]]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model, checkpoint = load_checkpoint_model(args.checkpoint, device)
    if checkpoint.get("normalization_stats") != dataset.normalization_stats:
        raise ValueError("Checkpoint normalization statistics differ from --stats-file")
    metrics = evaluate_loader(
        model,
        loader,
        dataset.normalization_stats,
        device,
        prediction_csv=args.predictions,
    )
    rendered = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(rendered)
    if args.metrics is not None:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

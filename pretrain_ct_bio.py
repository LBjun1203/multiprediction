#!/usr/bin/env python3
"""Train or preflight same-timepoint CT-biochemistry contrastive pretraining."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ct_modal.config import load_experiment_config
from evaluate import resolve_device
from pretraining.checkpoint import (
    build_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from pretraining.contrastive_model import CTBioContrastiveModel
from pretraining.losses import retrieval_recalls, symmetric_info_nce
from pretraining.paired_dataset import build_paired_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--ct-manifest", required=True, type=Path)
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--downstream-split", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--sheet", default="baseline")
    parser.add_argument("--ct-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ct_bio_pretrain"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_config_from_values(ct_spec: Any, values: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "ct_feature_dim",
        "ct_encoder_base_width",
        "ct_dropout",
        "bio_hidden_dim",
        "bio_feature_dim",
        "bio_dropout",
        "projection_hidden_dim",
        "projection_dim",
        "temperature",
        "learnable_temperature",
        "min_temperature",
        "max_temperature",
    )
    missing = sorted(set(keys) - set(values))
    if missing:
        raise ValueError(f"Pretraining config is missing model keys: {missing}")
    return {"ct_config": ct_spec.to_dict(), **{key: values[key] for key in keys}}


def validate_values(values: dict[str, Any]) -> None:
    if values.get("objective") != "contrastive":
        raise ValueError(
            "This entry point currently trains objective='contrastive'; the regression "
            "control model/loss are available but require a separately approved run config"
        )
    positive = (
        "batch_size",
        "epochs",
        "ct_learning_rate",
        "bio_learning_rate",
        "head_learning_rate",
        "gradient_clip",
    )
    for key in positive:
        if float(values.get(key, 0)) <= 0:
            raise ValueError(f"Pretraining config {key} must be positive")
    if int(values.get("batch_size", 0)) < 2:
        raise ValueError("Contrastive batch_size must be at least 2")
    if float(values.get("weight_decay", -1)) < 0:
        raise ValueError("weight_decay must be non-negative")
    if int(values.get("patience", -1)) < 0:
        raise ValueError("patience must be non-negative")


def make_optimizer(
    model: CTBioContrastiveModel, values: dict[str, Any]
) -> torch.optim.AdamW:
    head_parameters: list[torch.nn.Parameter] = [
        *model.ct_projector.parameters(),
        *model.bio_projector.parameters(),
    ]
    if isinstance(model.log_temperature, torch.nn.Parameter):
        head_parameters.append(model.log_temperature)
    return torch.optim.AdamW(
        [
            {"params": model.ct_encoder.parameters(), "lr": values["ct_learning_rate"]},
            {"params": model.bio_encoder.parameters(), "lr": values["bio_learning_rate"]},
            {"params": head_parameters, "lr": values["head_learning_rate"]},
        ],
        weight_decay=values["weight_decay"],
    )


def run_epoch(
    model: CTBioContrastiveModel,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    gradient_clip: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    pair_count = 0
    metric_sums: dict[str, float] = {}
    for raw_batch in loader:
        batch = {
            key: (
                value.to(device, non_blocking=device.type == "cuda")
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in raw_batch.items()
        }
        batch_size = int(batch["ct"].shape[0])
        if batch_size < 2:
            raise ValueError("InfoNCE loader produced a batch smaller than 2")
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                output = model(batch["ct"], batch["bio"], batch["bio_mask"])
                loss, logits = symmetric_info_nce(
                    output["z_ct"], output["z_bio"], output["temperature"]
                )
            if optimizer is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
        metrics = retrieval_recalls(logits.detach(), (1, 5))
        loss_sum += float(loss.item()) * batch_size
        pair_count += batch_size
        for key, value in metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + value * batch_size
    if pair_count == 0:
        raise ValueError("Pretraining loader produced no complete contrastive batches")
    return {
        "loss": loss_sum / pair_count,
        "pairs": float(pair_count),
        **{key: value / pair_count for key, value in metric_sums.items()},
    }


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    experiment = load_experiment_config(args.config)
    values = experiment.values
    validate_values(values)
    seed_everything(int(values["seed"]))
    bundle = build_paired_datasets(
        args.input,
        args.ct_manifest,
        args.split_file,
        experiment.ct,
        sheet_name=args.sheet,
        ct_root=args.ct_root,
        downstream_split_path=args.downstream_split,
    )
    print(json.dumps(bundle.preflight, ensure_ascii=False, indent=2))
    if args.preflight_only:
        return
    batch_size = int(values["batch_size"])
    if min(len(bundle.train), len(bundle.val)) < batch_size:
        raise ValueError(
            "batch_size exceeds a pretraining split; reduce it so validation has "
            "at least one complete InfoNCE batch"
        )
    device = resolve_device(args.device)
    use_amp = bool(values["amp"] and device.type == "cuda")
    train_loader = DataLoader(
        bundle.train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        bundle.val,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model_config = model_config_from_values(experiment.ct, values)
    start_epoch = 1
    history: list[dict[str, Any]] = []
    best_validation_metric = float("inf")
    epochs_without_improvement = 0
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if args.resume is not None:
        model, resumed = load_checkpoint(args.resume, map_location=device)
        if model.model_config != model_config:
            raise ValueError("Resume checkpoint model config differs from --config")
        optimizer = make_optimizer(model, values)
        optimizer.load_state_dict(resumed["optimizer_state"])
        scaler.load_state_dict(resumed["scaler_state"])
        start_epoch = int(resumed["epoch"]) + 1
        best_validation_metric = float(resumed["best_validation_metric"])
        history = list(resumed["history"])
    else:
        model = CTBioContrastiveModel.from_config(model_config).to(device)
        optimizer = make_optimizer(model, values)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        **values,
        "ct": experiment.ct.to_dict(),
        "data": {
            "input": str(args.input),
            "ct_manifest": str(args.ct_manifest),
            "split_file": str(args.split_file),
            "downstream_split": str(args.downstream_split) if args.downstream_split else None,
            "sheet": args.sheet,
        },
        "actual_batch_size": batch_size,
        "preflight": bundle.preflight,
        "normalization_provenance": bundle.normalization_stats.get("provenance"),
    }
    (args.output_dir / "resolved_config.json").write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    best_path = args.output_dir / "best.pt"
    last_path = args.output_dir / "last.pt"
    for epoch in range(start_epoch, int(values["epochs"]) + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            gradient_clip=float(values["gradient_clip"]),
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            scaler=scaler,
            use_amp=use_amp,
            gradient_clip=float(values["gradient_clip"]),
        )
        row = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        improved = val_metrics["loss"] < best_validation_metric
        if improved:
            best_validation_metric = val_metrics["loss"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        checkpoint = build_checkpoint(
            model,
            optimizer,
            scaler,
            epoch=epoch,
            best_validation_metric=best_validation_metric,
            normalization_stats=bundle.normalization_stats,
            split_provenance=bundle.preflight["split_provenance"],
            history=history,
        )
        save_checkpoint(last_path, checkpoint)
        if improved:
            save_checkpoint(best_path, checkpoint)
        (args.output_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if int(values["patience"]) and epochs_without_improvement >= int(values["patience"]):
            print("Early stopping: validation loss did not improve.")
            break


if __name__ == "__main__":
    main()

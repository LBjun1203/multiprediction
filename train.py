#!/usr/bin/env python3
"""Train the end-to-end Huashan multimodal forecasting model."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ct_modal.config import (
    ExperimentConfig,
    load_experiment_config,
    save_resolved_config,
)
from ct_preparation.manifest import (
    LEGACY_CT_INPUT_SPEC,
    resolve_ct_input_spec,
)
from evaluate import (
    evaluate_loader,
    load_checkpoint_model,
    move_batch_to_device,
    resolve_device,
)
from fusion_query_modal.fusion_time_query import masked_huber_loss
from multidataset.huashan_dataset import (
    ALL_BIOMARKERS,
    build_datasets,
    make_patient_balanced_sampler,
    normalization_values_equal,
)
from multimodal_model import HuashanForecastModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--ct-manifest", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        help="Experiment YAML; defaults to the legacy roi_25d contract when omitted",
    )
    parser.add_argument(
        "--split-file", type=Path, default=Path("multidataset/patient_split.csv")
    )
    parser.add_argument(
        "--stats-file", type=Path, default=Path("multidataset/normalization.json")
    )
    parser.add_argument("--sheet", default="model_long_0_2y")
    parser.add_argument("--ct-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/huashan"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--finetune-ct-backbone",
        action="store_true",
        help="Train the selected CT backbone; by default it is frozen",
    )
    parser.add_argument(
        "--reuse-stats",
        action="store_true",
        help="Reuse compatible normalization stats instead of fitting the final CT-eligible train cohort",
    )
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if state is None:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def train_one_epoch(
    model: HuashanForecastModel,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    grad_clip: float | None,
) -> float:
    model.train()
    weighted_loss_sum = 0.0
    supervised_count = 0.0
    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            prediction = model(batch)
            loss = masked_huber_loss(prediction, batch["target"], batch["target_mask"])
        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        valid_count = float(batch["target_mask"].sum().item())
        weighted_loss_sum += float(loss.item()) * valid_count
        supervised_count += valid_count
    if supervised_count <= 0:
        raise ValueError("Training loader produced no supervised entries")
    return weighted_loss_sum / supervised_count


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def save_checkpoint(
    path: Path,
    model: HuashanForecastModel,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_val_loss: float,
    normalization_stats: dict[str, Any],
    args: argparse.Namespace,
    history: list[dict[str, float | int]],
    epochs_without_improvement: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_config": dict(model.model_config),
            "biomarker_order": list(ALL_BIOMARKERS),
            "normalization_stats": normalization_stats,
            "training_args": serializable_args(args),
            "seed": args.seed,
            "rng_state": capture_rng_state(),
            "history": history,
            "epochs_without_improvement": epochs_without_improvement,
        },
        temporary,
    )
    temporary.replace(path)


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.epochs <= 0:
        raise ValueError("--batch-size and --epochs must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError(
            "--learning-rate must be positive and --weight-decay non-negative"
        )
    if args.patience < 0 or args.num_workers < 0:
        raise ValueError("--patience and --num-workers must be non-negative")
    if args.image_size < 32:
        raise ValueError("--image-size must be at least 32")
    if args.grad_clip is not None and args.grad_clip <= 0:
        raise ValueError("--grad-clip must be positive")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.config is not None:
        experiment = load_experiment_config(args.config)
        ct_spec = experiment.ct
        config_values = experiment.values
        cli_options = {
            argument.split("=", 1)[0]
            for argument in sys.argv[1:]
            if argument.startswith("--")
        }
        option_by_key = {
            "seed": "--seed",
            "batch_size": "--batch-size",
            "learning_rate": "--learning-rate",
            "weight_decay": "--weight-decay",
        }
        for key, option in option_by_key.items():
            if key in config_values and option not in cli_options:
                setattr(args, key, config_values[key])
        if "pretrained_ct" in config_values and "--no-pretrained" not in cli_options:
            args.no_pretrained = not bool(config_values["pretrained_ct"])
        if (
            "freeze_ct_backbone" in config_values
            and "--finetune-ct-backbone" not in cli_options
        ):
            args.finetune_ct_backbone = not bool(config_values["freeze_ct_backbone"])
    else:
        ct_spec = resolve_ct_input_spec(
            {
                **LEGACY_CT_INPUT_SPEC.to_dict(),
                "expected_shape": [3, args.image_size, args.image_size],
            }
        )
        experiment = ExperimentConfig(
            ct=ct_spec, values={}, source=Path("<cli-default>")
        )
    validate_args(args)
    resumed: tuple[HuashanForecastModel, dict[str, Any]] | None = None
    seed_everything(args.seed)
    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    if args.amp and not use_amp:
        print("AMP requested but CUDA is unavailable; continuing without AMP.")

    if args.resume is not None:
        resumed = load_checkpoint_model(args.resume, device)
        checkpoint_spec = resumed[0].ct_input_spec
        if args.config is not None and checkpoint_spec != ct_spec:
            raise ValueError(
                f"Resume checkpoint CT config {checkpoint_spec} differs from --config {ct_spec}"
            )
        ct_spec = checkpoint_spec

    train_dataset, val_dataset, test_dataset = build_datasets(
        xlsx_path=args.input,
        ct_manifest_path=args.ct_manifest,
        split_csv=args.split_file,
        stats_path=args.stats_file,
        sheet_name=args.sheet,
        fit_stats=not args.reuse_stats,
        ct_root=args.ct_root,
        ct_config=ct_spec,
    )
    sampler = make_patient_balanced_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = 1
    best_val_loss = float("inf")
    history: list[dict[str, float | int]] = []
    epochs_without_improvement = 0
    if resumed is not None:
        model, checkpoint = resumed
        optimizer = AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint["best_val_loss"])
        history = list(checkpoint.get("history", []))
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        restore_rng_state(checkpoint.get("rng_state"))
        checkpoint_stats = checkpoint.get("normalization_stats")
        if checkpoint_stats is None or not normalization_values_equal(
            checkpoint_stats, train_dataset.normalization_stats
        ):
            raise ValueError(
                "Resume checkpoint normalization differs from the current cohort"
            )
    else:
        model = HuashanForecastModel(
            pretrained_ct=not args.no_pretrained,
            freeze_ct_backbone=not args.finetune_ct_backbone,
            ct_config=ct_spec,
        ).to(device)
        optimizer = AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolved_values = {
        **experiment.values,
        "training_args": serializable_args(args),
        "normalization_metadata": train_dataset.normalization_stats.get("metadata"),
    }
    save_resolved_config(
        ExperimentConfig(ct=ct_spec, values=resolved_values, source=experiment.source),
        args.output_dir / "resolved_config.json",
    )
    best_path = args.output_dir / "best.pt"
    last_path = args.output_dir / "last.pt"

    print(
        f"device={device}; train={len(train_dataset)}; val={len(val_dataset)}; "
        f"test={len(test_dataset)}; CT_backbone_frozen={model.ct_encoder.backbone_frozen}"
    )
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            use_amp,
            args.grad_clip,
        )
        validation_metrics = evaluate_loader(
            model,
            validation_loader,
            train_dataset.normalization_stats,
            device,
        )
        val_loss = float(validation_metrics["masked_huber_loss"])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch:03d} train_loss={train_loss:.8f} val_loss={val_loss:.8f}")

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scaler,
                epoch,
                best_val_loss,
                train_dataset.normalization_stats,
                args,
                history,
                epochs_without_improvement,
            )
        else:
            epochs_without_improvement += 1
        save_checkpoint(
            last_path,
            model,
            optimizer,
            scaler,
            epoch,
            best_val_loss,
            train_dataset.normalization_stats,
            args,
            history,
            epochs_without_improvement,
        )
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        if args.patience and epochs_without_improvement >= args.patience:
            print(f"Early stopping after {args.patience} epochs without improvement.")
            break

    evaluation_checkpoint = best_path
    if not evaluation_checkpoint.is_file() and args.resume is not None:
        resumed_best = args.resume.parent / "best.pt"
        evaluation_checkpoint = resumed_best if resumed_best.is_file() else args.resume
    if not evaluation_checkpoint.is_file():
        raise FileNotFoundError("Training completed without producing a checkpoint")
    best_model, _ = load_checkpoint_model(evaluation_checkpoint, device)
    test_metrics = evaluate_loader(
        best_model,
        test_loader,
        train_dataset.normalization_stats,
        device,
        prediction_csv=args.output_dir / "test_predictions.csv",
    )
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"best_val_loss": best_val_loss, "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()

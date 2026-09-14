#!/usr/bin/env python3
"""Evaluate mean, persistence, and patient-balanced ridge baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multidataset.huashan_dataset import ALL_BIOMARKERS, build_datasets

from experiments.common import (
    TabularArrays,
    atomic_write_json,
    evaluate_prediction_arrays,
    masked_huber_numpy,
    patient_balanced_weights,
    tabular_arrays_from_dataset,
    write_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--ct-manifest", required=True, type=Path)
    parser.add_argument(
        "--split-file", type=Path, default=Path("multidataset/patient_split.csv")
    )
    parser.add_argument(
        "--stats-file",
        type=Path,
        default=Path("runs/comparisons/v1/shared/normalization.json"),
    )
    parser.add_argument("--sheet", default="model_long_0_2y")
    parser.add_argument("--ct-root", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/comparisons/v1")
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--fit-stats",
        action="store_true",
        help="Fit frozen normalization on the CT-eligible training cohort",
    )
    parser.add_argument(
        "--ridge-alphas",
        type=float,
        nargs="+",
        default=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace baseline result files already in OUTPUT_DIR",
    )
    return parser.parse_args()


def fit_ridge(
    features: np.ndarray,
    target: np.ndarray,
    sample_weight: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Fit weighted ridge with an unpenalized intercept."""
    if alpha < 0:
        raise ValueError("Ridge alpha must be non-negative")
    x = np.column_stack([np.ones(len(features)), features]).astype(np.float64)
    y = np.asarray(target, dtype=np.float64)
    weights = np.asarray(sample_weight, dtype=np.float64)
    if x.shape[0] != y.shape[0] or y.shape[0] != weights.shape[0]:
        raise ValueError("Ridge inputs have inconsistent row counts")
    if not np.all(weights > 0):
        raise ValueError("Ridge sample weights must be positive")
    root_weight = np.sqrt(weights)[:, None]
    x_weighted = x * root_weight
    y_weighted = y * root_weight[:, 0]
    penalty = np.eye(x.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    system = x_weighted.T @ x_weighted + penalty
    rhs = x_weighted.T @ y_weighted
    try:
        return np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(system, rhs, rcond=None)[0]


def predict_ridge(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    x = np.column_stack([np.ones(len(features)), features]).astype(np.float64)
    return x @ np.asarray(coefficients, dtype=np.float64)


def fit_multitarget_ridge(
    train: TabularArrays, alpha: float
) -> list[np.ndarray]:
    patient_weights = patient_balanced_weights(train.patient_ids)
    coefficients: list[np.ndarray] = []
    for marker_index in range(len(ALL_BIOMARKERS)):
        valid = train.target_mask[:, marker_index].astype(bool)
        coefficients.append(
            fit_ridge(
                train.features[valid],
                train.target[valid, marker_index],
                patient_weights[valid],
                alpha,
            )
        )
    return coefficients


def predict_multitarget_ridge(
    arrays: TabularArrays, coefficients: list[np.ndarray]
) -> np.ndarray:
    return np.column_stack(
        [predict_ridge(arrays.features, beta) for beta in coefficients]
    )


def choose_ridge_alpha(
    train: TabularArrays,
    validation: TabularArrays,
    alphas: list[float] | tuple[float, ...],
) -> tuple[float, list[np.ndarray], list[dict[str, float]]]:
    if not alphas:
        raise ValueError("At least one ridge alpha is required")
    trials: list[dict[str, float]] = []
    best: tuple[float, float, list[np.ndarray]] | None = None
    for alpha in sorted({float(value) for value in alphas}):
        coefficients = fit_multitarget_ridge(train, alpha)
        prediction = predict_multitarget_ridge(validation, coefficients)
        loss = masked_huber_numpy(
            prediction, validation.target, validation.target_mask
        )
        trials.append({"alpha": alpha, "validation_masked_huber_loss": loss})
        candidate = (loss, alpha, coefficients)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    assert best is not None
    return best[1], best[2], trials


def persistence_prediction(
    arrays: TabularArrays, normalization_stats: dict[str, Any]
) -> np.ndarray:
    """Express baseline clinical values in each target's standardized space."""
    result = np.empty_like(arrays.target, dtype=np.float64)
    for marker_index, marker in enumerate(ALL_BIOMARKERS):
        values = arrays.dataframe[f"base_{marker}"].to_numpy(float)
        if marker == "CysC":
            base_stats = normalization_stats["base"][marker]
            observed = arrays.base_mask[:, marker_index].astype(bool)
            values = np.where(observed, values, float(base_stats["median"]))
        target_stats = normalization_stats["target"][marker]
        result[:, marker_index] = (
            values - float(target_stats["mean"])
        ) / float(target_stats["std"])
    if not np.isfinite(result).all():
        raise ValueError("Persistence prediction contains non-finite values")
    return result


def save_result(
    root: Path,
    name: str,
    prediction: np.ndarray,
    arrays: TabularArrays,
    stats: dict[str, Any],
    evaluation_mask: np.ndarray | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics, frame = evaluate_prediction_arrays(
        prediction, arrays, stats, evaluation_mask=evaluation_mask
    )
    metrics["model"] = name
    metrics["evaluation_mask"] = (
        "target_and_baseline_observed" if evaluation_mask is not None else "target"
    )
    if extra:
        metrics.update(extra)
    write_evaluation(root / name, metrics, frame)
    return metrics


def ensure_fresh_outputs(args: argparse.Namespace) -> None:
    protected = [
        args.output_dir / name / "test_metrics.json"
        for name in ("mean", "persistence_imputed", "persistence_observed", "ridge")
    ]
    if args.fit_stats:
        protected.append(args.stats_file)
    existing = [str(path) for path in protected if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing baseline artifacts: "
            f"{existing}. Choose a new output/stats path or pass --overwrite."
        )


def main() -> None:
    args = parse_args()
    if args.image_size < 32:
        raise ValueError("--image-size must be at least 32")
    if any(value < 0 for value in args.ridge_alphas):
        raise ValueError("--ridge-alphas must be non-negative")
    ensure_fresh_outputs(args)

    train_ds, val_ds, test_ds = build_datasets(
        xlsx_path=args.input,
        ct_manifest_path=args.ct_manifest,
        split_csv=args.split_file,
        stats_path=args.stats_file,
        sheet_name=args.sheet,
        fit_stats=args.fit_stats,
        expected_ct_shape=(3, args.image_size, args.image_size),
        ct_root=args.ct_root,
    )
    train = tabular_arrays_from_dataset(train_ds)
    validation = tabular_arrays_from_dataset(val_ds)
    test = tabular_arrays_from_dataset(test_ds)
    stats = train_ds.normalization_stats
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}
    results["mean"] = save_result(
        args.output_dir, "mean", np.zeros_like(test.target), test, stats
    )

    persistence = persistence_prediction(test, stats)
    results["persistence_imputed"] = save_result(
        args.output_dir,
        "persistence_imputed",
        persistence,
        test,
        stats,
        extra={"cysc_baseline_policy": "training_median_if_missing"},
    )
    observed_mask = test.target_mask.astype(bool) & test.base_mask.astype(bool)
    results["persistence_observed"] = save_result(
        args.output_dir,
        "persistence_observed",
        persistence,
        test,
        stats,
        evaluation_mask=observed_mask,
        extra={"cysc_baseline_policy": "evaluate_only_when_observed"},
    )

    best_alpha, coefficients, trials = choose_ridge_alpha(
        train, validation, args.ridge_alphas
    )
    ridge_prediction = predict_multitarget_ridge(test, coefficients)
    results["ridge"] = save_result(
        args.output_dir,
        "ridge",
        ridge_prediction,
        test,
        stats,
        extra={"selected_alpha": best_alpha},
    )
    atomic_write_json(
        {
            "selection_metric": "validation_masked_huber_loss",
            "selected_alpha": best_alpha,
            "trials": trials,
        },
        args.output_dir / "ridge" / "validation_selection.json",
    )
    atomic_write_json(
        {
            "input": str(args.input.resolve()),
            "ct_manifest": str(args.ct_manifest.resolve()),
            "split_file": str(args.split_file.resolve()),
            "stats_file": str(args.stats_file.resolve()),
            "fit_stats": bool(args.fit_stats),
            "ridge_alphas": [float(value) for value in args.ridge_alphas],
            "feature_order": [
                *[f"base_{name}_z" for name in ALL_BIOMARKERS],
                *[f"base_{name}_mask" for name in ALL_BIOMARKERS],
                "age_z",
                "sex_code",
                "time_years",
            ],
            "cohort": {
                "train_observations": len(train.patient_ids),
                "validation_observations": len(validation.patient_ids),
                "test_observations": len(test.patient_ids),
                "train_patients": len(np.unique(train.patient_ids)),
                "validation_patients": len(np.unique(validation.patient_ids)),
                "test_patients": len(np.unique(test.patient_ids)),
            },
        },
        args.output_dir / "classical_manifest.json",
    )
    summary = {
        name: {
            "masked_huber_loss": values["masked_huber_loss"],
            "normalized_rmse": values["macro_average"]["normalized_rmse"],
            "positive_r_squared": values["positive_r_squared"],
        }
        for name, values in results.items()
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate classical, Tabular, and Full test outputs into one comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multidataset.huashan_dataset import ALL_BIOMARKERS

from experiments.common import (
    atomic_write_csv,
    atomic_write_json,
    enrich_metrics_from_prediction_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("runs/comparisons/v1")
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Defaults to ROOT/summary"
    )
    return parser.parse_args()


def cohort_signature(frame: pd.DataFrame) -> str:
    required = {"patient_id", "time_years"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {missing}")
    normalized = frame.loc[:, ["patient_id", "time_years"]].copy()
    normalized["patient_id"] = normalized["patient_id"].astype(str)
    normalized["time_years"] = normalized["time_years"].astype(float).map(
        lambda value: f"{value:.10f}"
    )
    payload = normalized.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def result_name(root: Path, metric_path: Path, metrics: dict[str, Any]) -> str:
    relative = metric_path.parent.relative_to(root)
    model = metrics.get("model")
    seed = metrics.get("seed")
    if model:
        return f"{model}_seed{seed}" if seed is not None else str(model)
    parts = list(relative.parts)
    if parts and parts[0].lower() in {"full", "tabular"}:
        return "_".join(parts)
    return "_".join(parts) or "result"


def load_results(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    metric_paths = sorted(
        path
        for path in root.rglob("test_metrics.json")
        if "summary" not in path.relative_to(root).parts
    )
    for metric_path in metric_paths:
        prediction_path = metric_path.with_name("test_predictions.csv")
        if not prediction_path.is_file():
            print(f"Skipping {metric_path}: no sibling test_predictions.csv")
            continue
        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        predictions = pd.read_csv(prediction_path)
        enriched = enrich_metrics_from_prediction_csv(metrics, predictions)
        records.append(
            {
                "name": result_name(root, metric_path, enriched),
                "metrics": enriched,
                "metrics_path": metric_path,
                "prediction_path": prediction_path,
                "cohort_signature": cohort_signature(predictions),
            }
        )
    names = [record["name"] for record in records]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"Duplicate result labels: {duplicates}")
    if not records:
        raise FileNotFoundError(f"No complete test results found under {root}")
    return records


def build_tables(
    records: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    overall_rows: list[dict[str, Any]] = []
    marker_rows: list[dict[str, Any]] = []
    for record in records:
        name = record["name"]
        metrics = record["metrics"]
        overall_rows.append(
            {
                "model": name,
                "evaluation_mask": metrics.get("evaluation_mask", "target"),
                "observations": metrics.get("observations"),
                "patients": metrics.get("patients"),
                "supervised_elements": metrics.get("supervised_elements"),
                "masked_huber_loss": metrics.get("masked_huber_loss"),
                "macro_normalized_mae": metrics["macro_average"].get(
                    "normalized_mae"
                ),
                "macro_normalized_rmse": metrics["macro_average"].get(
                    "normalized_rmse"
                ),
                "macro_clinical_mae": metrics["macro_average"].get("clinical_mae"),
                "macro_clinical_rmse": metrics["macro_average"].get(
                    "clinical_rmse"
                ),
                "positive_r_squared": metrics.get("positive_r_squared"),
                "cohort_signature": record["cohort_signature"],
                "metrics_path": str(record["metrics_path"].resolve()),
            }
        )
        for marker in ALL_BIOMARKERS:
            values = metrics["per_biomarker"][marker]
            marker_rows.append(
                {
                    "model": name,
                    "biomarker": marker,
                    **values,
                }
            )
    signatures = sorted({record["cohort_signature"] for record in records})
    audit = {
        "same_ordered_test_rows": len(signatures) == 1,
        "cohort_signatures": signatures,
        "results": {
            record["name"]: {
                "signature": record["cohort_signature"],
                "metrics": str(record["metrics_path"].resolve()),
                "predictions": str(record["prediction_path"].resolve()),
            }
            for record in records
        },
        "note": (
            "persistence_observed uses a stricter CysC mask; compare its aggregate "
            "metrics with care even when ordered test rows match"
        ),
    }
    return pd.DataFrame(overall_rows), pd.DataFrame(marker_rows), audit


def plot_comparison(marker_table: pd.DataFrame, output_path: Path) -> None:
    models = marker_table["model"].drop_duplicates().tolist()
    markers = list(ALL_BIOMARKERS)
    x = np.arange(len(markers), dtype=float)
    width = min(0.8 / max(len(models), 1), 0.24)
    figure, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(models), 1)))

    for model_index, (model, color) in enumerate(zip(models, colors, strict=True)):
        subset = marker_table.loc[marker_table["model"] == model].set_index(
            "biomarker"
        )
        offset = (model_index - (len(models) - 1) / 2) * width
        rmse = [subset.loc[marker, "normalized_rmse"] for marker in markers]
        r_squared = [subset.loc[marker, "r_squared"] for marker in markers]
        axes[0].bar(x + offset, rmse, width=width, label=model, color=color)
        axes[1].bar(x + offset, r_squared, width=width, label=model, color=color)

    axes[0].set_ylabel("Normalized RMSE (lower is better)")
    axes[0].set_title("Held-out test comparison on the shared CT-eligible cohort")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=min(4, len(models)), fontsize=8)
    axes[1].set_ylabel("R² (higher is better)")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_xticks(x, markers, rotation=35, ha="right")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.root / "summary"
    records = load_results(args.root)
    overall, per_biomarker, audit = build_tables(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(overall, output_dir / "comparison_overall.csv")
    atomic_write_csv(per_biomarker, output_dir / "comparison_per_biomarker.csv")
    atomic_write_json(audit, output_dir / "cohort_audit.json")
    plot_comparison(per_biomarker, output_dir / "comparison_metrics.png")
    print(overall.to_string(index=False))
    if not audit["same_ordered_test_rows"]:
        raise RuntimeError(
            "Result files do not contain the same ordered test rows; see cohort_audit.json"
        )


if __name__ == "__main__":
    main()

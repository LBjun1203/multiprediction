#!/usr/bin/env python3
"""Plot training curves and test metrics from a Huashan run directory.

Expected input files:
  history.json
  test_metrics.json
  test_predictions.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


REQUIRED_FILES = ("history.json", "test_metrics.json", "test_predictions.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs"),
        help="Directory containing the three run files (default: runs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: <run-dir>/plots)",
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def configure_style() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    candidates = (
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    )
    selected = next((name for name in candidates if name in available), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [selected],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_inputs(run_dir: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing files in {run_dir.resolve()}: {', '.join(missing)}"
        )


def load_history(path: Path) -> pd.DataFrame:
    raw = read_json(path)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"history.json must contain a non-empty list: {path}")
    frame = pd.DataFrame(raw)
    required = {"epoch", "train_loss", "val_loss"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"history.json is missing keys: {sorted(missing)}")
    frame = frame.loc[:, ["epoch", "train_loss", "val_loss"]].apply(
        pd.to_numeric, errors="raise"
    )
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("history.json contains non-finite values")
    return frame.sort_values("epoch", kind="stable").reset_index(drop=True)


def load_test_results(
    metrics_path: Path, predictions_path: Path
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    metrics = read_json(metrics_path)
    per_marker = metrics.get("per_biomarker")
    if not isinstance(per_marker, dict) or not per_marker:
        raise ValueError("test_metrics.json has no per_biomarker metrics")
    predictions = pd.read_csv(predictions_path)

    rows: list[dict[str, float | int | str]] = []
    for marker, values in per_marker.items():
        required_columns = {
            f"{marker}_mask",
            f"{marker}_prediction",
            f"{marker}_target",
        }
        missing = required_columns - set(predictions.columns)
        if missing:
            raise ValueError(
                f"test_predictions.csv is missing columns for {marker}: "
                f"{sorted(missing)}"
            )

        valid = predictions[f"{marker}_mask"].eq(1) & predictions[
            f"{marker}_target"
        ].notna()
        prediction = predictions.loc[valid, f"{marker}_prediction"].to_numpy(float)
        target = predictions.loc[valid, f"{marker}_target"].to_numpy(float)
        if prediction.size == 0:
            pearson_r = np.nan
            r_squared = np.nan
        else:
            pearson_r = (
                float(np.corrcoef(prediction, target)[0, 1])
                if prediction.size > 1
                and np.std(prediction) > 0
                and np.std(target) > 0
                else np.nan
            )
            denominator = float(np.sum((target - target.mean()) ** 2))
            r_squared = (
                float(1.0 - np.sum((prediction - target) ** 2) / denominator)
                if denominator > 0
                else np.nan
            )

        rows.append(
            {
                "marker": marker,
                "count": int(values["count"]),
                "normalized_mae": float(values["normalized_mae"]),
                "normalized_rmse": float(values["normalized_rmse"]),
                "clinical_mae": float(values["clinical_mae"]),
                "clinical_rmse": float(values["clinical_rmse"]),
                "pearson_r": pearson_r,
                "r_squared": r_squared,
            }
        )
    return metrics, predictions, pd.DataFrame(rows)


def plot_loss_curves(
    history: pd.DataFrame,
    test_loss: float,
    output_path: Path,
    dpi: int,
) -> None:
    best_index = int(history["val_loss"].idxmin())
    best = history.loc[best_index]
    final = history.iloc[-1]

    fig, axis = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    axis.plot(
        history["epoch"],
        history["train_loss"],
        color="#2563EB",
        linewidth=2.0,
        label="Train loss",
    )
    axis.plot(
        history["epoch"],
        history["val_loss"],
        color="#F97316",
        linewidth=2.0,
        label="Validation loss",
    )
    axis.axhline(
        test_loss,
        color="#7C3AED",
        linewidth=1.5,
        linestyle=":",
        label=f"Test loss = {test_loss:.4f}",
    )
    axis.axvline(
        best["epoch"], color="#16A34A", linewidth=1.4, linestyle="--"
    )
    axis.scatter(
        [best["epoch"]],
        [best["val_loss"]],
        color="#16A34A",
        s=55,
        zorder=5,
    )
    axis.annotate(
        f"Best validation\nEpoch {int(best['epoch'])}, loss {best['val_loss']:.4f}",
        xy=(best["epoch"], best["val_loss"]),
        xytext=(16, -52),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#16A34A"},
        fontsize=10,
    )
    axis.set_title("Training and Validation Loss", fontsize=16, fontweight="bold")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Masked Huber loss")
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.65)
    axis.legend(frameon=False, ncols=3, loc="upper right")
    axis.text(
        0.01,
        0.02,
        (
            f"Final epoch: train={final['train_loss']:.4f}, "
            f"val={final['val_loss']:.4f}. "
            f"Best validation occurred {int(final['epoch'] - best['epoch'])} "
            "epochs before training ended."
        ),
        transform=axis.transAxes,
        fontsize=9.5,
        color="#4B5563",
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_test_metrics(
    metrics: dict[str, Any],
    marker_frame: pd.DataFrame,
    output_path: Path,
    dpi: int,
) -> None:
    count = len(marker_frame)
    y = np.arange(count)
    height = 0.34

    fig = plt.figure(figsize=(15.5, 11.0))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(1.0, 0.82),
        left=0.06,
        right=0.98,
        bottom=0.05,
        top=0.85,
        hspace=0.38,
        wspace=0.18,
    )
    error_axis = fig.add_subplot(grid[0, 0])
    quality_axis = fig.add_subplot(grid[0, 1])
    table_axis = fig.add_subplot(grid[1, :])

    error_axis.barh(
        y - height / 2,
        marker_frame["normalized_mae"],
        height=height,
        color="#2563EB",
        label="Normalized MAE",
    )
    error_axis.barh(
        y + height / 2,
        marker_frame["normalized_rmse"],
        height=height,
        color="#F97316",
        label="Normalized RMSE",
    )
    error_axis.set_yticks(y, marker_frame["marker"])
    error_axis.invert_yaxis()
    error_axis.set_xlabel("Standardized error (lower is better)")
    error_axis.set_title("Per-biomarker standardized errors", fontweight="bold")
    error_axis.grid(axis="x", color="#D1D5DB", linewidth=0.7, alpha=0.65)
    error_axis.legend(frameon=False, loc="lower right")

    r2_colors = np.where(
        marker_frame["r_squared"] >= 0, "#16A34A", "#DC2626"
    )
    quality_axis.barh(
        y,
        marker_frame["r_squared"],
        height=0.52,
        color=r2_colors,
        alpha=0.82,
        label="R-squared",
    )
    quality_axis.scatter(
        marker_frame["pearson_r"],
        y,
        color="#7C3AED",
        edgecolor="white",
        linewidth=0.8,
        s=55,
        zorder=4,
        label="Pearson r",
    )
    quality_axis.axvline(0.0, color="#111827", linewidth=1.0)
    quality_axis.set_yticks(y, marker_frame["marker"])
    quality_axis.invert_yaxis()
    quality_axis.set_xlabel("Coefficient")
    quality_axis.set_title("Prediction quality", fontweight="bold")
    quality_axis.grid(axis="x", color="#D1D5DB", linewidth=0.7, alpha=0.65)
    quality_axis.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        ncols=2,
    )

    table_axis.axis("off")
    table_columns = [
        "Biomarker",
        "N",
        "Clinical MAE",
        "Clinical RMSE",
        "Pearson r",
        "R-squared",
    ]
    table_rows = [
        [
            row.marker,
            f"{int(row.count)}",
            f"{row.clinical_mae:.3f}",
            f"{row.clinical_rmse:.3f}",
            f"{row.pearson_r:.3f}" if np.isfinite(row.pearson_r) else "NA",
            f"{row.r_squared:.3f}" if np.isfinite(row.r_squared) else "NA",
        ]
        for row in marker_frame.itertuples(index=False)
    ]
    table = table_axis.table(
        cellText=table_rows,
        colLabels=table_columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=(0.02, 0.02, 0.96, 0.92),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        cell.set_linewidth(0.6)
        if row_index == 0:
            cell.set_facecolor("#E5E7EB")
            cell.set_text_props(weight="bold", color="#111827")
        else:
            cell.set_facecolor("#FFFFFF" if row_index % 2 else "#F9FAFB")
            if column_index == 5:
                value = marker_frame.iloc[row_index - 1]["r_squared"]
                cell.set_text_props(color="#15803D" if value >= 0 else "#B91C1C")
    table_axis.set_title(
        "Clinical-unit metrics (interpret within each biomarker's own unit)",
        fontweight="bold",
        pad=6,
    )

    macro = metrics["macro_average"]
    positive_r2 = int((marker_frame["r_squared"] > 0).sum())
    fig.suptitle(
        "Independent Test-set Performance",
        fontsize=17,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.925,
        (
            f"Observations: {int(metrics['observations'])}    |    "
            f"Masked Huber loss: {float(metrics['masked_huber_loss']):.4f}    |    "
            f"Macro normalized MAE / RMSE: "
            f"{float(macro['normalized_mae']):.3f} / "
            f"{float(macro['normalized_rmse']):.3f}    |    "
            f"Positive R-squared: {positive_r2}/{count}"
        ),
        ha="center",
        fontsize=10.5,
        color="#374151",
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")

    run_dir = args.run_dir.expanduser().resolve()
    validate_inputs(run_dir)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    history = load_history(run_dir / "history.json")
    metrics, _, marker_frame = load_test_results(
        run_dir / "test_metrics.json", run_dir / "test_predictions.csv"
    )

    loss_path = output_dir / "loss_curves.png"
    metrics_path = output_dir / "test_metrics.png"
    plot_loss_curves(
        history,
        float(metrics["masked_huber_loss"]),
        loss_path,
        args.dpi,
    )
    plot_test_metrics(metrics, marker_frame, metrics_path, args.dpi)

    print(f"Saved: {loss_path}")
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()

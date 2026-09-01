#!/usr/bin/env python3
"""Create a reproducible patient-level train/validation/test split."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SHEET = "model_long_0_2y"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a Huashan longitudinal table at patient level."
    )
    parser.add_argument("--input", required=True, type=Path, help="Cleaned .xlsx file")
    parser.add_argument("--sheet", default=DEFAULT_SHEET, help="Input sheet name")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("multidataset/patient_split.csv"),
        help="Output CSV path",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def clean_patient_id(value: object) -> str:
    """Return a stripped string ID without Excel's numeric ``.0`` suffix."""
    if pd.isna(value):
        raise ValueError("patient_id contains a missing value")
    text = str(value).strip()
    if not text:
        raise ValueError("patient_id contains an empty value")
    match = re.fullmatch(r"(\d+)\.0", text)
    return match.group(1) if match else text


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if not np.isfinite(ratios).all() or (ratios <= 0).any():
        raise ValueError("train/val/test ratios must all be finite and greater than 0")
    if not np.isclose(ratios.sum(), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError(
            f"train/val/test ratios must sum to 1.0; got {ratios.sum():.12g}"
        )


def allocate_split_counts(patient_count: int, ratios: np.ndarray) -> np.ndarray:
    """Allocate integer counts with the largest-remainder method."""
    raw_counts = ratios * patient_count
    counts = np.floor(raw_counts).astype(int)
    remainder = patient_count - int(counts.sum())
    fractional = raw_counts - counts
    order = np.argsort(-fractional, kind="stable")
    counts[order[:remainder]] += 1
    if (counts == 0).any():
        raise ValueError(
            f"At least three patients are required for non-empty splits; got {patient_count}"
        )
    return counts


def read_split_source(xlsx_path: Path, sheet_name: str) -> pd.DataFrame:
    if not xlsx_path.is_file():
        raise FileNotFoundError(f"Input Excel file does not exist: {xlsx_path}")

    df = pd.read_excel(
        xlsx_path,
        sheet_name=sheet_name,
        dtype={"patient_id": "string"},
    )
    required = {"patient_id", "t_days"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Sheet {sheet_name!r} is missing columns: {missing}")
    if df.empty:
        raise ValueError(f"Sheet {sheet_name!r} has no observations")

    df = df.copy()
    df["patient_id"] = df["patient_id"].map(clean_patient_id)
    df["t_days"] = pd.to_numeric(df["t_days"], errors="coerce")
    invalid_time = (
        df["t_days"].isna()
        | ~np.isfinite(df["t_days"])
        | (df["t_days"] <= 0)
        | (df["t_days"] > 730)
    )
    if invalid_time.any():
        examples = df.loc[invalid_time, ["patient_id", "t_days"]].head(10)
        raise ValueError(
            "t_days must satisfy 0 < t_days <= 730 for every observation. "
            f"Invalid examples:\n{examples.to_string(index=False)}"
        )
    return df


def create_patient_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """Assign each unique patient to exactly one split."""
    validate_ratios(train_ratio, val_ratio, test_ratio)
    all_ids = sorted(set(df["patient_id"]))
    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    counts = allocate_split_counts(len(all_ids), ratios)

    shuffled_ids = np.asarray(all_ids, dtype=object)
    np.random.default_rng(seed).shuffle(shuffled_ids)
    n_train, n_val, _ = counts.tolist()
    train_ids = set(shuffled_ids[:n_train].tolist())
    val_ids = set(shuffled_ids[n_train : n_train + n_val].tolist())
    test_ids = set(shuffled_ids[n_train + n_val :].tolist())
    all_patient_ids = set(all_ids)

    # Splitting is patient-level even though each table row is one observation.
    assert train_ids & val_ids == set()
    assert train_ids & test_ids == set()
    assert val_ids & test_ids == set()
    assert train_ids | val_ids | test_ids == all_patient_ids

    assignments = {
        patient_id: split
        for split, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids))
        for patient_id in ids
    }
    return pd.DataFrame(
        {
            "patient_id": all_ids,
            "split": [assignments[patient_id] for patient_id in all_ids],
        }
    )


def print_split_summary(df: pd.DataFrame, split_df: pd.DataFrame) -> None:
    split_by_patient = split_df.set_index("patient_id")["split"]
    observation_split = df["patient_id"].map(split_by_patient)
    if observation_split.isna().any():
        raise AssertionError("Some observations have no patient split assignment")

    counts = split_df["split"].value_counts()
    print(f"All patients: {len(split_df)}")
    print(f"Train patients: {int(counts.get('train', 0))}")
    print(f"Val patients: {int(counts.get('val', 0))}")
    print(f"Test patients: {int(counts.get('test', 0))}")
    print()

    for split in ("train", "val", "test"):
        split_rows = df.loc[observation_split == split]
        patient_count = int(split_rows["patient_id"].nunique())
        observation_count = len(split_rows)
        median_days = float(split_rows["t_days"].median())
        year_0_1 = int(split_rows["t_days"].between(0, 365, inclusive="right").sum())
        year_1_2 = int(
            ((split_rows["t_days"] > 365) & (split_rows["t_days"] <= 730)).sum()
        )
        print(f"{split.capitalize()} observations: {observation_count}")
        print(f"  patients: {patient_count}")
        print(
            f"  mean observations per patient: {observation_count / patient_count:.2f}"
        )
        print(f"  median t_days: {median_days:.2f}")
        print(f"  0-1y observations: {year_0_1}")
        print(f"  1-2y observations: {year_1_2}")


def main() -> None:
    args = parse_args()
    df = read_split_source(args.input, args.sheet)
    split_df = create_patient_split(
        df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(args.output, index=False, encoding="utf-8")
    print_split_summary(df, split_df)
    print(f"\nSaved patient split: {args.output}")


if __name__ == "__main__":
    main()

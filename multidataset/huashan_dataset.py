#!/usr/bin/env python3
"""Unified CT, biochemistry, demographics, time, and target Dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

CORE_BIOMARKERS = [
    "HBA1C",
    "TG",
    "CHO",
    "HDL",
    "LDL",
    "ALB",
    "BUN",
    "UA",
    "CRE",
]
ALL_BIOMARKERS = CORE_BIOMARKERS + ["CysC"]
DEFAULT_SHEET = "model_long_0_2y"
EPSILON = 1e-12
CT_SUCCESS_STATUSES = {"success", "skipped_existing"}
DEFAULT_CT_SHAPE = (3, 224, 224)

BASE_COLUMNS = [f"base_{name}" for name in ALL_BIOMARKERS]
TARGET_COLUMNS = [f"target_{name}" for name in ALL_BIOMARKERS]
NUMERIC_COLUMNS = (
    BASE_COLUMNS
    + ["base_CysC_mask"]
    + TARGET_COLUMNS
    + ["target_CysC_mask", "age", "sex_code", "t_days"]
)
REQUIRED_COLUMNS = ["patient_id", "ct_path"] + NUMERIC_COLUMNS
TENSOR_KEYS = (
    "ct_25d",
    "bio_base",
    "base_mask",
    "demographics",
    "time",
    "target",
    "target_mask",
)

CT_MANIFEST_REQUIRED_COLUMNS = (
    "patient_id",
    "npy_25d_path",
    "build_25d_status",
)


def clean_patient_id(value: object) -> str:
    if pd.isna(value):
        raise ValueError("patient_id contains a missing value")
    text = str(value).strip()
    if not text:
        raise ValueError("patient_id contains an empty value")
    match = re.fullmatch(r"(\d+)\.0", text)
    return match.group(1) if match else text


def _first_bad_row(df: pd.DataFrame, mask: pd.Series, column: str) -> str:
    row = df.loc[mask, ["patient_id", column]].iloc[0]
    return f"patient_id={row['patient_id']!r}, column={column!r}, value={row[column]!r}"


def _validate_dataframe(df: pd.DataFrame, context: str) -> None:
    if df.empty:
        raise ValueError(f"{context} has no observations")

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"{context} is missing required columns: {missing_columns}")

    empty_ct = df["ct_path"].isna() | df["ct_path"].astype(
        str
    ).str.strip().str.lower().isin({"", "nan", "none", "<na>"})
    if empty_ct.any():
        raise ValueError(
            _first_bad_row(df, empty_ct, "ct_path") + ": ct_path must be non-empty"
        )

    for column in [f"base_{name}" for name in CORE_BIOMARKERS] + [
        f"target_{name}" for name in CORE_BIOMARKERS
    ]:
        values = pd.to_numeric(df[column], errors="coerce")
        invalid = values.isna() | ~np.isfinite(values.to_numpy(dtype=np.float64))
        if invalid.any():
            raise ValueError(
                _first_bad_row(df, pd.Series(invalid, index=df.index), column)
                + ": core biomarkers cannot be missing or non-finite"
            )

    for column in ("age", "sex_code", "t_days"):
        values = pd.to_numeric(df[column], errors="coerce")
        invalid = values.isna() | ~np.isfinite(values.to_numpy(dtype=np.float64))
        if invalid.any():
            raise ValueError(
                _first_bad_row(df, pd.Series(invalid, index=df.index), column)
                + ": value must be present and finite"
            )

    invalid_sex = ~df["sex_code"].isin([0, 1])
    if invalid_sex.any():
        raise ValueError(
            _first_bad_row(df, invalid_sex, "sex_code") + ": sex_code must be 0 or 1"
        )
    invalid_time = (df["t_days"] <= 0) | (df["t_days"] > 730)
    if invalid_time.any():
        raise ValueError(
            _first_bad_row(df, invalid_time, "t_days") + ": expected 0 < t_days <= 730"
        )

    for value_column, mask_column in (
        ("base_CysC", "base_CysC_mask"),
        ("target_CysC", "target_CysC_mask"),
    ):
        invalid_mask = df[mask_column].isna() | ~df[mask_column].isin([0, 1])
        if invalid_mask.any():
            raise ValueError(
                _first_bad_row(df, invalid_mask, mask_column)
                + ": mask must contain only 0 or 1"
            )

        values = pd.to_numeric(df[value_column], errors="coerce")
        observed = df[mask_column] == 1
        invalid_observed = observed & (
            values.isna() | ~np.isfinite(values.fillna(0).to_numpy(dtype=np.float64))
        )
        if invalid_observed.any():
            raise ValueError(
                _first_bad_row(df, invalid_observed, value_column)
                + f": {mask_column}=1 requires a finite value"
            )
        inconsistent_missing = (df[mask_column] == 0) & values.notna()
        if inconsistent_missing.any():
            raise ValueError(
                _first_bad_row(df, inconsistent_missing, value_column)
                + f": {mask_column}=0 requires a missing value"
            )


def load_longitudinal_table(
    xlsx_path: str | Path,
    sheet_name: str = DEFAULT_SHEET,
) -> pd.DataFrame:
    path = Path(xlsx_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input Excel file does not exist: {path}")
    df = pd.read_excel(path, sheet_name=sheet_name, dtype={"patient_id": "string"})
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"Sheet {sheet_name!r} is missing required columns: {missing_columns}"
        )

    df = df.copy()
    df["patient_id"] = df["patient_id"].map(clean_patient_id)
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    _validate_dataframe(df, f"sheet {sheet_name!r}")
    return df.reset_index(drop=True)


def load_patient_split(split_csv: str | Path) -> pd.DataFrame:
    path = Path(split_csv)
    if not path.is_file():
        raise FileNotFoundError(f"Patient split file does not exist: {path}")
    split_df = pd.read_csv(path, dtype={"patient_id": "string", "split": "string"})
    required = {"patient_id", "split"}
    missing = sorted(required - set(split_df.columns))
    if missing:
        raise ValueError(f"Split CSV is missing columns: {missing}")

    split_df = split_df.loc[:, ["patient_id", "split"]].copy()
    split_df["patient_id"] = split_df["patient_id"].map(clean_patient_id)
    split_df["split"] = split_df["split"].astype(str).str.strip().str.lower()
    if split_df["patient_id"].duplicated().any():
        duplicate = split_df.loc[
            split_df["patient_id"].duplicated(), "patient_id"
        ].iloc[0]
        raise ValueError(f"Split CSV has duplicate patient_id: {duplicate!r}")
    invalid = ~split_df["split"].isin(["train", "val", "test"])
    if invalid.any():
        raise ValueError(
            f"Invalid split value: {split_df.loc[invalid, 'split'].iloc[0]!r}"
        )
    present = set(split_df["split"])
    if present != {"train", "val", "test"}:
        raise ValueError(
            f"Split CSV must contain train, val, and test; found {sorted(present)}"
        )
    return split_df


def load_ct_manifest(
    manifest_csv: str | Path,
    ct_root: str | Path | None = None,
    required_patient_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Load CT metadata and validate files used by the current cohort only.

    Relative ``npy_25d_path`` values are resolved against ``ct_root`` when it is
    provided, otherwise against the manifest directory. Absolute paths remain
    supported for backward compatibility.
    """
    path = Path(manifest_csv)
    if not path.is_file():
        raise FileNotFoundError(f"CT manifest does not exist: {path}")
    manifest = pd.read_csv(path, dtype={"patient_id": "string"}, keep_default_na=False)
    missing = sorted(set(CT_MANIFEST_REQUIRED_COLUMNS) - set(manifest.columns))
    if missing:
        raise ValueError(f"CT manifest is missing required columns: {missing}")

    manifest = manifest.copy()
    manifest["patient_id"] = manifest["patient_id"].map(clean_patient_id)
    manifest["build_25d_status"] = (
        manifest["build_25d_status"].astype(str).str.strip().str.lower()
    )
    manifest["npy_25d_path"] = manifest["npy_25d_path"].astype(str).str.strip()
    if manifest["patient_id"].duplicated().any():
        duplicate = manifest.loc[
            manifest["patient_id"].duplicated(), "patient_id"
        ].iloc[0]
        raise ValueError(f"CT manifest has duplicate patient_id: {duplicate!r}")

    usable = manifest["build_25d_status"].isin(CT_SUCCESS_STATUSES)
    validate_rows = usable
    if required_patient_ids is not None:
        validate_rows = validate_rows & manifest["patient_id"].isin(
            required_patient_ids
        )
    empty_path = manifest["npy_25d_path"].str.lower().isin({"", "nan", "none", "<na>"})
    if (validate_rows & empty_path).any():
        patient_id = manifest.loc[validate_rows & empty_path, "patient_id"].iloc[0]
        raise ValueError(
            f"CT manifest marks patient_id={patient_id!r} successful but npy_25d_path is empty"
        )
    base_directory = Path(ct_root) if ct_root is not None else path.parent
    for index in manifest.index[usable & ~empty_path]:
        array_path = Path(str(manifest.loc[index, "npy_25d_path"]))
        if not array_path.is_absolute():
            array_path = base_directory / array_path
        manifest.loc[index, "npy_25d_path"] = str(array_path.resolve())

    for row in manifest.loc[validate_rows, ["patient_id", "npy_25d_path"]].itertuples(
        index=False
    ):
        if not Path(row.npy_25d_path).is_file():
            raise FileNotFoundError(
                f"CT manifest marks patient_id={row.patient_id!r} successful but the array "
                f"does not exist: {row.npy_25d_path}"
            )
    return manifest


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _patient_id_hash(patient_ids: set[str]) -> str:
    payload = "\n".join(sorted(patient_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_normalization_metadata(
    train_df: pd.DataFrame,
    split_csv: str | Path,
    ct_manifest_path: str | Path,
) -> dict[str, Any]:
    patient_ids = set(train_df["patient_id"].astype(str))
    return {
        "schema_version": 2,
        "cohort_definition": "ct_eligible_train",
        "train_patient_count": len(patient_ids),
        "train_observation_count": len(train_df),
        "train_patient_ids_sha256": _patient_id_hash(patient_ids),
        "split_file_sha256": _sha256_file(split_csv),
        "ct_manifest_sha256": _sha256_file(ct_manifest_path),
    }


def validate_normalization_metadata(
    stats: dict[str, Any], expected_metadata: dict[str, Any]
) -> None:
    metadata = stats.get("metadata")
    if metadata is None:
        warnings.warn(
            "Normalization statistics have no cohort metadata. They are accepted for "
            "backward compatibility, but should be regenerated with --fit-stats.",
            stacklevel=2,
        )
        return
    mismatches = {
        key: (metadata.get(key), expected_value)
        for key, expected_value in expected_metadata.items()
        if metadata.get(key) != expected_value
    }
    if mismatches:
        details = "; ".join(
            f"{key}: stored={stored!r}, current={current!r}"
            for key, (stored, current) in mismatches.items()
        )
        raise ValueError(
            "Normalization statistics do not match the current train/CT cohort. "
            f"Regenerate them with --fit-stats. {details}"
        )


def merge_longitudinal_with_ct(
    longitudinal_df: pd.DataFrame, ct_manifest: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Attach one baseline CT to every usable follow-up observation."""
    usable_manifest = ct_manifest.loc[
        ct_manifest["build_25d_status"].isin(CT_SUCCESS_STATUSES),
        ["patient_id", "npy_25d_path"],
    ].copy()
    all_patient_ids = set(longitudinal_df["patient_id"])
    usable_ct_ids = set(usable_manifest["patient_id"])
    merged = longitudinal_df.merge(
        usable_manifest,
        on="patient_id",
        how="inner",
        validate="many_to_one",
    )
    summary = {
        "biochemistry_patients": len(all_patient_ids),
        "biochemistry_observations": len(longitudinal_df),
        "manifest_patients": len(ct_manifest),
        "usable_ct_patients": len(usable_ct_ids),
        "missing_or_failed_ct_patients": len(all_patient_ids - usable_ct_ids),
        "ct_patients_without_biochemistry": len(usable_ct_ids - all_patient_ids),
        "multimodal_patients": int(merged["patient_id"].nunique()),
        "multimodal_observations": len(merged),
    }
    if merged.empty:
        raise ValueError("No observations remain after joining successful CT patients")
    return merged.reset_index(drop=True), summary


def load_ct_tensor(
    npy_path: str | Path,
    expected_shape: tuple[int, int, int] = DEFAULT_CT_SHAPE,
) -> torch.Tensor:
    """Read one deterministic [C,H,W] CT array and enforce its storage contract."""
    path = Path(npy_path)
    if not path.is_file():
        raise FileNotFoundError(f"2.5D CT array does not exist: {path}")
    array = np.load(path, allow_pickle=False)
    if array.shape != expected_shape:
        raise ValueError(
            f"2.5D CT array has shape {array.shape}, expected {expected_shape}: {path}"
        )
    if array.dtype != np.float32:
        raise ValueError(f"2.5D CT array must be float32, got {array.dtype}: {path}")
    if not np.isfinite(array).all():
        raise ValueError(f"2.5D CT array contains non-finite values: {path}")
    minimum = float(array.min())
    maximum = float(array.max())
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(
            f"2.5D CT array must be in [0, 1], got [{minimum}, {maximum}]: {path}"
        )
    return torch.from_numpy(np.ascontiguousarray(array)).to(dtype=torch.float32)


def _safe_mean_std(series: pd.Series, label: str) -> tuple[float, float]:
    values = pd.to_numeric(series, errors="coerce").dropna().astype(np.float64)
    if values.empty or not np.isfinite(values.to_numpy()).all():
        raise ValueError(
            f"Cannot fit {label}: no finite values or non-finite values present"
        )
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    if not np.isfinite(std) or abs(std) < EPSILON:
        std = 1.0
    return mean, std


def fit_normalization_stats(train_df: pd.DataFrame) -> dict[str, Any]:
    """Fit every statistic from train patients/observations only."""
    _validate_dataframe(train_df, "training dataframe")

    # Baseline values repeat in the long table. Give every train patient one vote.
    train_baseline_df = (
        train_df.sort_values(["patient_id", "t_days"], kind="stable")
        .drop_duplicates(subset="patient_id", keep="first")
        .reset_index(drop=True)
    )

    base_stats: dict[str, dict[str, float]] = {}
    for name in CORE_BIOMARKERS:
        mean, std = _safe_mean_std(train_baseline_df[f"base_{name}"], f"base_{name}")
        base_stats[name] = {"mean": mean, "std": std}

    observed_base_cysc = train_baseline_df.loc[
        train_baseline_df["base_CysC_mask"] == 1, "base_CysC"
    ]
    if observed_base_cysc.empty:
        raise ValueError(
            "Cannot fit base_CysC: no observed CysC among unique train patients"
        )
    base_cysc_median = float(observed_base_cysc.median())
    filled_base_cysc = train_baseline_df["base_CysC"].fillna(base_cysc_median)
    base_cysc_mean, base_cysc_std = _safe_mean_std(filled_base_cysc, "base_CysC")
    base_stats["CysC"] = {
        "median": base_cysc_median,
        "mean": base_cysc_mean,
        "std": base_cysc_std,
    }

    target_stats: dict[str, dict[str, float]] = {}
    for name in CORE_BIOMARKERS:
        mean, std = _safe_mean_std(train_df[f"target_{name}"], f"target_{name}")
        target_stats[name] = {"mean": mean, "std": std}

    observed_target_cysc = train_df.loc[
        train_df["target_CysC_mask"] == 1, "target_CysC"
    ]
    if observed_target_cysc.empty:
        raise ValueError(
            "Cannot fit target_CysC: no observed CysC in train observations"
        )
    target_cysc_mean, target_cysc_std = _safe_mean_std(
        observed_target_cysc, "target_CysC"
    )
    target_stats["CysC"] = {"mean": target_cysc_mean, "std": target_cysc_std}

    age_mean, age_std = _safe_mean_std(train_baseline_df["age"], "age")
    return {
        "biomarker_order": ALL_BIOMARKERS,
        "base": base_stats,
        "target": target_stats,
        "age": {"mean": age_mean, "std": age_std},
    }


def save_normalization_stats(stats: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(stats, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")


def _validate_stats(stats: dict[str, Any]) -> None:
    if stats.get("biomarker_order") != ALL_BIOMARKERS:
        raise ValueError(
            "normalization biomarker_order does not match the required fixed order: "
            f"{ALL_BIOMARKERS}"
        )
    for section in ("base", "target"):
        for name in ALL_BIOMARKERS:
            entry = stats.get(section, {}).get(name, {})
            for key in ("mean", "std"):
                value = entry.get(key)
                if not isinstance(value, (int, float)) or not np.isfinite(value):
                    raise ValueError(
                        f"Invalid normalization statistic: {section}.{name}.{key}"
                    )
            if entry["std"] <= 0:
                raise ValueError(
                    f"Normalization std must be positive: {section}.{name}.std"
                )
    median = stats.get("base", {}).get("CysC", {}).get("median")
    if not isinstance(median, (int, float)) or not np.isfinite(median):
        raise ValueError("Invalid normalization statistic: base.CysC.median")
    for key in ("mean", "std"):
        value = stats.get("age", {}).get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"Invalid normalization statistic: age.{key}")
    if stats["age"]["std"] <= 0:
        raise ValueError("Normalization std must be positive: age.std")


def load_normalization_stats(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Normalization file does not exist: {source}. Run with --fit-stats first."
        )
    with source.open("r", encoding="utf-8") as file:
        stats = json.load(file)
    _validate_stats(stats)
    return stats


class HuashanMultimodalDataset(Dataset):
    """Observation-level multimodal Dataset with one baseline CT per patient."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        normalization_stats: dict[str, Any],
        expected_ct_shape: tuple[int, int, int] = DEFAULT_CT_SHAPE,
    ):
        _validate_stats(normalization_stats)
        _validate_dataframe(dataframe, "dataset dataframe")
        if "npy_25d_path" not in dataframe.columns:
            raise ValueError("dataset dataframe is missing npy_25d_path")
        if len(expected_ct_shape) != 3 or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in expected_ct_shape
        ):
            raise ValueError(
                f"expected_ct_shape must contain three positive integers, got {expected_ct_shape}"
            )
        if expected_ct_shape[0] != 3:
            raise ValueError(
                "The current CTEncoder requires exactly 3 CT slices; "
                f"got expected_ct_shape={expected_ct_shape}"
            )
        if min(expected_ct_shape[1:]) < 32:
            raise ValueError(
                "The current CTEncoder requires spatial dimensions of at least 32; "
                f"got expected_ct_shape={expected_ct_shape}"
            )
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.normalization_stats = normalization_stats
        self.expected_ct_shape = expected_ct_shape
        self.patient_ids = self.dataframe["patient_id"].astype(str).tolist()

    def __len__(self) -> int:
        # One Dataset item is one follow-up observation, not one patient.
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, str | torch.Tensor]:
        row = self.dataframe.iloc[index]
        patient_id = str(row["patient_id"])
        ct_25d_path = str(row["npy_25d_path"])

        base_mask_values = [1.0] * len(CORE_BIOMARKERS) + [float(row["base_CysC_mask"])]
        target_mask_values = [1.0] * len(CORE_BIOMARKERS) + [
            float(row["target_CysC_mask"])
        ]

        base_values: list[float] = []
        for name in CORE_BIOMARKERS:
            raw = float(row[f"base_{name}"])
            entry = self.normalization_stats["base"][name]
            base_values.append((raw - entry["mean"]) / entry["std"])
        base_cysc_stats = self.normalization_stats["base"]["CysC"]
        base_cysc = (
            float(row["base_CysC"])
            if row["base_CysC_mask"] == 1
            else float(base_cysc_stats["median"])
        )
        base_values.append(
            (base_cysc - base_cysc_stats["mean"]) / base_cysc_stats["std"]
        )

        target_values: list[float] = []
        for name in CORE_BIOMARKERS:
            raw = float(row[f"target_{name}"])
            entry = self.normalization_stats["target"][name]
            target_values.append((raw - entry["mean"]) / entry["std"])
        if row["target_CysC_mask"] == 1:
            target_cysc_stats = self.normalization_stats["target"]["CysC"]
            target_values.append(
                (float(row["target_CysC"]) - target_cysc_stats["mean"])
                / target_cysc_stats["std"]
            )
        else:
            # This is a placeholder in standardized space and is ignored by the mask.
            target_values.append(0.0)

        age_stats = self.normalization_stats["age"]
        tensors = {
            "ct_25d": load_ct_tensor(ct_25d_path, self.expected_ct_shape),
            "bio_base": torch.tensor(base_values, dtype=torch.float32),
            "base_mask": torch.tensor(base_mask_values, dtype=torch.float32),
            "demographics": torch.tensor(
                [
                    (float(row["age"]) - age_stats["mean"]) / age_stats["std"],
                    float(row["sex_code"]),
                ],
                dtype=torch.float32,
            ),
            "time": torch.tensor([float(row["t_days"]) / 365.25], dtype=torch.float32),
            "target": torch.tensor(target_values, dtype=torch.float32),
            "target_mask": torch.tensor(target_mask_values, dtype=torch.float32),
        }

        for key, tensor in tensors.items():
            if not torch.isfinite(tensor).all():
                raise ValueError(
                    f"Non-finite tensor for patient_id={patient_id!r}, key={key!r}"
                )
        for key in ("base_mask", "target_mask"):
            if not torch.all((tensors[key] == 0) | (tensors[key] == 1)):
                raise ValueError(
                    f"Non-binary mask for patient_id={patient_id!r}, key={key!r}"
                )
        if not torch.all(tensors["base_mask"][:9] == 1):
            raise ValueError(f"Invalid core base mask for patient_id={patient_id!r}")
        if not torch.all(tensors["target_mask"][:9] == 1):
            raise ValueError(f"Invalid core target mask for patient_id={patient_id!r}")

        return {
            "patient_id": patient_id,
            "source_ct_path": str(row["ct_path"]),
            "ct_25d_path": ct_25d_path,
            **tensors,
        }


def make_patient_balanced_sampler(
    dataset: HuashanMultimodalDataset,
) -> WeightedRandomSampler:
    """Give each observation weight 1 / patient_observation_count."""
    if len(dataset) == 0:
        raise ValueError("Cannot build a sampler for an empty dataset")
    counts = Counter(dataset.patient_ids)
    weights = torch.tensor(
        [1.0 / counts[patient_id] for patient_id in dataset.patient_ids],
        dtype=torch.double,
    )
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)


def build_datasets(
    xlsx_path: str | Path,
    ct_manifest_path: str | Path,
    split_csv: str | Path,
    stats_path: str | Path,
    sheet_name: str = DEFAULT_SHEET,
    fit_stats: bool = False,
    expected_ct_shape: tuple[int, int, int] = DEFAULT_CT_SHAPE,
    ct_root: str | Path | None = None,
) -> tuple[
    HuashanMultimodalDataset,
    HuashanMultimodalDataset,
    HuashanMultimodalDataset,
]:
    full_df = load_longitudinal_table(xlsx_path, sheet_name)
    split_df = load_patient_split(split_csv)

    data_ids = set(full_df["patient_id"])
    split_ids = set(split_df["patient_id"])
    missing_assignments = sorted(data_ids - split_ids)
    unknown_assignments = sorted(split_ids - data_ids)
    if missing_assignments or unknown_assignments:
        raise ValueError(
            "Patient IDs in data and split CSV differ. "
            f"Missing assignments (first 10): {missing_assignments[:10]}; "
            f"unknown assignments (first 10): {unknown_assignments[:10]}"
        )

    split_map = split_df.set_index("patient_id")["split"]
    working_df = full_df.copy()
    working_df["split"] = working_df["patient_id"].map(split_map)
    ct_manifest = load_ct_manifest(
        ct_manifest_path,
        ct_root=ct_root,
        required_patient_ids=data_ids,
    )
    working_df, ct_summary = merge_longitudinal_with_ct(working_df, ct_manifest)
    print("Multimodal join summary:")
    for key, value in ct_summary.items():
        print(f"  {key}: {value}")
    print()

    frames = {
        name: working_df.loc[working_df["split"] == name]
        .drop(columns="split")
        .reset_index(drop=True)
        for name in ("train", "val", "test")
    }
    empty_splits = [name for name, frame in frames.items() if frame.empty]
    if empty_splits:
        raise ValueError(
            f"No CT-eligible observations remain in splits: {empty_splits}"
        )

    id_sets = {name: set(frame["patient_id"]) for name, frame in frames.items()}
    assert id_sets["train"] & id_sets["val"] == set()
    assert id_sets["train"] & id_sets["test"] == set()
    assert id_sets["val"] & id_sets["test"] == set()
    assert set().union(*id_sets.values()) == set(working_df["patient_id"])

    expected_metadata = build_normalization_metadata(
        frames["train"], split_csv, ct_manifest_path
    )
    if fit_stats:
        stats = fit_normalization_stats(frames["train"])
        stats["metadata"] = expected_metadata
        save_normalization_stats(stats, stats_path)
    else:
        stats = load_normalization_stats(stats_path)
        validate_normalization_metadata(stats, expected_metadata)

    return tuple(
        HuashanMultimodalDataset(frames[name], stats, expected_ct_shape)
        for name in ("train", "val", "test")
    )


def _dataset_summary(name: str, dataset: HuashanMultimodalDataset) -> str:
    return (
        f"{name} dataset: {len(dataset)} observations / "
        f"{dataset.dataframe['patient_id'].nunique()} patients"
    )


def run_smoke_test(
    train_dataset: HuashanMultimodalDataset,
    val_dataset: HuashanMultimodalDataset,
    test_dataset: HuashanMultimodalDataset,
) -> None:
    print(_dataset_summary("Train", train_dataset))
    print(_dataset_summary("Val", val_dataset))
    print(_dataset_summary("Test", test_dataset))
    print()

    expected_shapes = {
        "ct_25d": train_dataset.expected_ct_shape,
        "bio_base": (10,),
        "base_mask": (10,),
        "demographics": (2,),
        "time": (1,),
        "target": (10,),
        "target_mask": (10,),
    }
    for dataset in (train_dataset, val_dataset, test_dataset):
        for index in range(len(dataset)):
            item = dataset[index]
            for key, expected_shape in expected_shapes.items():
                tensor = item[key]
                assert isinstance(tensor, torch.Tensor)
                assert tensor.shape == expected_shape
                assert tensor.dtype == torch.float32
                assert torch.isfinite(tensor).all()
            assert torch.all((item["base_mask"] == 0) | (item["base_mask"] == 1))
            assert torch.all((item["target_mask"] == 0) | (item["target_mask"] == 1))
            assert torch.all(item["base_mask"][:9] == 1)
            assert torch.all(item["target_mask"][:9] == 1)

    example = train_dataset[0]
    for key in expected_shapes:
        print(f"{key} shape: {example[key].shape}")
    print()
    print(f"Example patient_id: {example['patient_id']}")
    print(f"Example source_ct_path: {example['source_ct_path']}")
    print(f"Example ct_25d_path: {example['ct_25d_path']}")
    print(
        f"Example ct_25d range: [{example['ct_25d'].min().item():.6f}, "
        f"{example['ct_25d'].max().item():.6f}]"
    )
    print(f"Example t_years: {example['time'].item():.6f}")
    print()

    for name, dataset in (
        ("Train", train_dataset),
        ("Val", val_dataset),
        ("Test", test_dataset),
    ):
        base_missing = int((dataset.dataframe["base_CysC_mask"] == 0).sum())
        target_missing = int((dataset.dataframe["target_CysC_mask"] == 0).sum())
        print(f"{name} CysC missing: baseline={base_missing}, target={target_missing}")
    print(
        "Smoke test passed: CT, biochemistry, demographics, time, targets, and masks "
        "have valid shapes and finite values."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build unified Huashan multimodal PyTorch datasets."
    )
    parser.add_argument("--input", required=True, type=Path, help="Cleaned .xlsx file")
    parser.add_argument(
        "--ct-manifest",
        required=True,
        type=Path,
        help="CT preprocessing ct_manifest.csv",
    )
    parser.add_argument("--sheet", default=DEFAULT_SHEET, help="Input sheet name")
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path("multidataset/patient_split.csv"),
    )
    parser.add_argument(
        "--stats-file",
        type=Path,
        default=Path("multidataset/normalization.json"),
    )
    parser.add_argument("--num-slices", type=int, default=DEFAULT_CT_SHAPE[0])
    parser.add_argument("--image-size", type=int, default=DEFAULT_CT_SHAPE[1])
    parser.add_argument(
        "--ct-root",
        type=Path,
        help="Root used to resolve relative npy_25d_path values in the CT manifest",
    )
    parser.add_argument(
        "--fit-stats",
        action="store_true",
        help="Fit and save normalization using train observations/patients only",
    )
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_slices != 3:
        raise ValueError("The current end-to-end model requires --num-slices 3")
    if args.image_size < 32:
        raise ValueError("--image-size must be at least 32")
    train_dataset, val_dataset, test_dataset = build_datasets(
        xlsx_path=args.input,
        ct_manifest_path=args.ct_manifest,
        split_csv=args.split_file,
        stats_path=args.stats_file,
        sheet_name=args.sheet,
        fit_stats=args.fit_stats,
        expected_ct_shape=(args.num_slices, args.image_size, args.image_size),
        ct_root=args.ct_root,
    )
    if args.fit_stats:
        print(f"Saved train-only normalization statistics: {args.stats_file}\n")
    if args.smoke_test:
        run_smoke_test(train_dataset, val_dataset, test_dataset)
    else:
        print(_dataset_summary("Train", train_dataset))
        print(_dataset_summary("Val", val_dataset))
        print(_dataset_summary("Test", test_dataset))


if __name__ == "__main__":
    main()

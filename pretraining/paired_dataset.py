"""Patient-level same-timepoint CT-biochemistry Dataset and preflight helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ct_preparation.manifest import (
    REPRESENTATIONS,
    SUCCESS_STATUSES,
    CTInputSpec,
    load_manifest,
    resolve_ct_input_spec,
    resolve_representation_paths,
)
from multidataset.huashan_dataset import (
    ALL_BIOMARKERS,
    clean_patient_id,
    load_ct_tensor,
)

EPSILON = 1e-12


@dataclass(frozen=True)
class PairedDataBundle:
    train: "PairedCTBiochemistryDataset"
    val: "PairedCTBiochemistryDataset"
    normalization_stats: dict[str, Any]
    preflight: dict[str, Any]


def load_same_timepoint_table(
    path: str | Path, sheet_name: str = "baseline"
) -> pd.DataFrame:
    """Load one intended baseline CT-biochemistry row per patient."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Same-timepoint table does not exist: {source}")
    if source.suffix.lower() == ".csv":
        frame = pd.read_csv(source, dtype={"patient_id": "string"})
    else:
        frame = pd.read_excel(
            source, sheet_name=sheet_name, dtype={"patient_id": "string"}
        )
    return canonicalize_pair_table(frame)


def canonicalize_pair_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate columns and normalize masks without filling model values."""
    if frame.empty:
        raise ValueError("Same-timepoint table has no rows")
    required = {"patient_id", *(f"base_{name}" for name in ALL_BIOMARKERS)}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Same-timepoint table is missing columns: {missing}")
    result = frame.copy()
    result["patient_id"] = result["patient_id"].map(clean_patient_id)
    duplicate_mask = result["patient_id"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate = result.loc[duplicate_mask, "patient_id"].iloc[0]
        raise ValueError(
            "Pretraining requires one intended pair per patient; duplicate "
            f"patient_id={duplicate!r}"
        )

    for name in ALL_BIOMARKERS:
        value_column = f"base_{name}"
        mask_column = f"base_{name}_mask"
        values = pd.to_numeric(result[value_column], errors="coerce")
        finite = pd.Series(
            np.isfinite(values.fillna(0).to_numpy(dtype=np.float64)),
            index=result.index,
        )
        if mask_column in result.columns:
            masks = pd.to_numeric(result[mask_column], errors="coerce")
            invalid_mask = masks.isna() | ~masks.isin([0, 1])
            if invalid_mask.any():
                patient_id = result.loc[invalid_mask, "patient_id"].iloc[0]
                raise ValueError(
                    f"{mask_column} must contain 0 or 1 for patient_id={patient_id!r}"
                )
            observed = masks.eq(1)
            if (observed & (values.isna() | ~finite)).any():
                patient_id = result.loc[
                    observed & (values.isna() | ~finite), "patient_id"
                ].iloc[0]
                raise ValueError(
                    f"{mask_column}=1 requires finite {value_column} for "
                    f"patient_id={patient_id!r}"
                )
            if ((~observed) & values.notna()).any():
                patient_id = result.loc[(~observed) & values.notna(), "patient_id"].iloc[0]
                raise ValueError(
                    f"{mask_column}=0 requires missing {value_column} for "
                    f"patient_id={patient_id!r}"
                )
            result[mask_column] = masks.astype(np.int8)
        else:
            result[mask_column] = (values.notna() & finite).astype(np.int8)
        result[value_column] = values

    mask_columns = [f"base_{name}_mask" for name in ALL_BIOMARKERS]
    no_measurements = result[mask_columns].sum(axis=1).eq(0)
    if no_measurements.any():
        patient_id = result.loc[no_measurements, "patient_id"].iloc[0]
        raise ValueError(
            f"Patient {patient_id!r} has no observed biochemical measurements"
        )
    return result.reset_index(drop=True)


def fit_paired_normalization(frame: pd.DataFrame) -> dict[str, Any]:
    """Fit per-biomarker mean/std and fill values from train patients only."""
    frame = canonicalize_pair_table(frame)
    base: dict[str, dict[str, float]] = {}
    for name in ALL_BIOMARKERS:
        values = frame.loc[
            frame[f"base_{name}_mask"].eq(1), f"base_{name}"
        ].astype(np.float64)
        if values.empty or not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"Cannot fit base_{name}: no finite train observations")
        median = float(values.median())
        mean = float(values.mean())
        std = float(values.std(ddof=1))
        if not np.isfinite(std) or abs(std) < EPSILON:
            std = 1.0
        base[name] = {"mean": mean, "std": std, "median": median}
    return {"biomarker_order": list(ALL_BIOMARKERS), "base": base}


def validate_paired_normalization(stats: dict[str, Any]) -> None:
    if stats.get("biomarker_order") != list(ALL_BIOMARKERS):
        raise ValueError("Paired normalization biomarker order is incompatible")
    for name in ALL_BIOMARKERS:
        entry = stats.get("base", {}).get(name, {})
        for key in ("mean", "std", "median"):
            value = entry.get(key)
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"Invalid paired normalization: base.{name}.{key}")
        if entry["std"] <= 0:
            raise ValueError(f"Paired normalization std must be positive: {name}")


class PairedCTBiochemistryDataset(Dataset):
    """One fixed CT and one same-timepoint biochemical vector per patient."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        normalization_stats: dict[str, Any],
        ct_config: CTInputSpec | dict[str, Any],
        *,
        forbidden_patient_ids: set[str] | None = None,
    ) -> None:
        frame = canonicalize_pair_table(dataframe)
        validate_paired_normalization(normalization_stats)
        self.ct_input_spec = resolve_ct_input_spec(ct_config)
        if self.ct_input_spec.representation != "roi_3d":
            raise ValueError("First pretraining Dataset supports roi_3d only")
        if "ct_path_resolved" not in frame.columns:
            record = REPRESENTATIONS[self.ct_input_spec.representation]
            if record.path_column not in frame.columns:
                raise ValueError("Paired dataframe is missing ct_path_resolved")
            frame["ct_path_resolved"] = frame[record.path_column]
        forbidden = {clean_patient_id(value) for value in (forbidden_patient_ids or set())}
        overlap = sorted(set(frame["patient_id"]) & forbidden)
        if overlap:
            raise ValueError(
                "Pretraining split overlaps downstream validation/test patients: "
                f"{overlap[:10]}"
            )
        self.dataframe = frame.reset_index(drop=True)
        self.normalization_stats = normalization_stats
        self.patient_ids = self.dataframe["patient_id"].tolist()

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, str | torch.Tensor]:
        row = self.dataframe.iloc[index]
        patient_id = str(row["patient_id"])
        values: list[float] = []
        masks: list[float] = []
        for name in ALL_BIOMARKERS:
            observed = int(row[f"base_{name}_mask"])
            entry = self.normalization_stats["base"][name]
            raw = float(row[f"base_{name}"]) if observed else float(entry["median"])
            values.append((raw - float(entry["mean"])) / float(entry["std"]))
            masks.append(float(observed))
        bio = torch.tensor(values, dtype=torch.float32)
        bio_mask = torch.tensor(masks, dtype=torch.float32)
        ct_path = str(row["ct_path_resolved"])
        ct = load_ct_tensor(
            ct_path,
            expected_shape=self.ct_input_spec.expected_shape,
            representation=self.ct_input_spec.representation,
        )
        if not torch.isfinite(bio).all():
            raise ValueError(f"Non-finite biochemistry tensor for patient_id={patient_id!r}")
        return {
            "ct": ct,
            "bio": bio,
            "bio_mask": bio_mask,
            "patient_id": patient_id,
            "ct_path": ct_path,
        }


def _load_pretraining_split(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    split = pd.read_csv(source, dtype={"patient_id": "string", "split": "string"})
    missing = sorted({"patient_id", "split"} - set(split.columns))
    if missing:
        raise ValueError(f"Pretraining split is missing columns: {missing}")
    split = split.loc[:, ["patient_id", "split"]].copy()
    split["patient_id"] = split["patient_id"].map(clean_patient_id)
    split["split"] = split["split"].astype(str).str.strip().str.lower()
    if split["patient_id"].duplicated().any():
        raise ValueError("Pretraining split has duplicate patient_id")
    if not set(split["split"]).issubset({"train", "val"}):
        raise ValueError("Pretraining split values must be train or val")
    if set(split["split"]) != {"train", "val"}:
        raise ValueError("Pretraining split must contain train and val patients")
    return split


def _downstream_forbidden_ids(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    split = pd.read_csv(path, dtype={"patient_id": "string", "split": "string"})
    missing = sorted({"patient_id", "split"} - set(split.columns))
    if missing:
        raise ValueError(f"Downstream split is missing columns: {missing}")
    split["patient_id"] = split["patient_id"].map(clean_patient_id)
    split["split"] = split["split"].astype(str).str.strip().str.lower()
    invalid = ~split["split"].isin({"train", "val", "test"})
    if invalid.any():
        raise ValueError("Downstream split contains an invalid split value")
    return set(split.loc[split["split"].isin({"val", "test"}), "patient_id"])


def _sha256_payload(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_paired_datasets(
    table_path: str | Path,
    ct_manifest_path: str | Path,
    split_path: str | Path,
    ct_config: CTInputSpec | dict[str, Any],
    *,
    sheet_name: str = "baseline",
    ct_root: str | Path | None = None,
    downstream_split_path: str | Path | None = None,
    check_files: bool = True,
) -> PairedDataBundle:
    """Build leakage-aware train/validation paired Datasets and provenance."""
    spec = resolve_ct_input_spec(ct_config)
    if spec.representation != "roi_3d":
        raise ValueError("Same-timepoint pretraining currently requires roi_3d")
    pairs = load_same_timepoint_table(table_path, sheet_name)
    split = _load_pretraining_split(split_path)
    pair_ids = set(pairs["patient_id"])
    split_ids = set(split["patient_id"])
    if pair_ids != split_ids:
        raise ValueError(
            "Same-timepoint table and pretraining split patient IDs differ; "
            f"missing={sorted(pair_ids - split_ids)[:10]}, "
            f"unknown={sorted(split_ids - pair_ids)[:10]}"
        )

    forbidden = _downstream_forbidden_ids(downstream_split_path)
    overlap_before_exclusion = pair_ids & forbidden
    pairs = pairs.loc[~pairs["patient_id"].isin(forbidden)].copy()
    split = split.loc[~split["patient_id"].isin(forbidden)].copy()
    if set(split["split"]) != {"train", "val"}:
        raise ValueError(
            "Downstream validation/test exclusion leaves an empty pretraining split"
        )

    manifest_path = Path(ct_manifest_path)
    manifest = load_manifest(manifest_path)
    root = Path(ct_root) if ct_root is not None else manifest_path.parent
    resolved = resolve_representation_paths(
        manifest,
        spec,
        root,
        required_patient_ids=set(pairs["patient_id"]),
        check_files=check_files,
    )
    record = REPRESENTATIONS[spec.representation]
    usable = resolved.loc[
        resolved[record.status_column]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(SUCCESS_STATUSES),
        ["patient_id", record.path_column],
    ].rename(columns={record.path_column: "ct_path_resolved"})
    merged = pairs.merge(usable, on="patient_id", how="inner", validate="one_to_one")
    merged = merged.merge(split, on="patient_id", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("No CT-eligible same-timepoint pairs remain")
    frames = {
        name: merged.loc[merged["split"].eq(name)].drop(columns="split").reset_index(drop=True)
        for name in ("train", "val")
    }
    if any(frame.empty for frame in frames.values()):
        raise ValueError("CT eligibility leaves an empty pretraining train/val split")
    normalization = fit_paired_normalization(frames["train"])
    train_ids = sorted(frames["train"]["patient_id"].tolist())
    val_ids = sorted(frames["val"]["patient_id"].tolist())
    normalization["provenance"] = {
        "schema_version": 1,
        "fit_split": "train",
        "train_patient_count": len(train_ids),
        "train_patient_ids_sha256": _sha256_payload(train_ids),
    }
    missingness = {
        name: {
            "observed": int(merged[f"base_{name}_mask"].sum()),
            "missing": int((merged[f"base_{name}_mask"] == 0).sum()),
        }
        for name in ALL_BIOMARKERS
    }
    preflight = {
        "input_pairs": len(pair_ids),
        "unique_patients": len(pair_ids),
        "duplicate_patient_count": 0,
        "excluded_downstream_val_test_patients": len(overlap_before_exclusion),
        "ct_eligible_pairs": len(merged),
        "ct_eligibility_failures": len(pairs) - len(merged),
        "train_pairs": len(frames["train"]),
        "val_pairs": len(frames["val"]),
        "train_downstream_val_test_overlap": len(set(train_ids) & forbidden),
        "val_downstream_val_test_overlap": len(set(val_ids) & forbidden),
        "missingness": missingness,
        "split_provenance": {
            "train_patient_ids_sha256": _sha256_payload(train_ids),
            "val_patient_ids_sha256": _sha256_payload(val_ids),
            "forbidden_patient_ids_sha256": _sha256_payload(sorted(forbidden)),
        },
    }
    return PairedDataBundle(
        train=PairedCTBiochemistryDataset(
            frames["train"], normalization, spec, forbidden_patient_ids=forbidden
        ),
        val=PairedCTBiochemistryDataset(
            frames["val"], normalization, spec, forbidden_patient_ids=forbidden
        ),
        normalization_stats=normalization,
        preflight=preflight,
    )

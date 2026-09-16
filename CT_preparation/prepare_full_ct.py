#!/usr/bin/env python3
"""Prepare one canonical, unwindowed ``full_ct`` NIfTI per patient."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from ct_preparation.common import (
    clean_patient_id,
    orient_lps,
    patient_directory,
    relative_manifest_path,
)
from ct_preparation.dicom_series import read_series, select_series
from ct_preparation.manifest import canonicalize_manifest, load_manifest, write_manifest


def load_input_table(path: Path, sheet_name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input Excel file does not exist: {path}")
    table = pd.read_excel(path, sheet_name=sheet_name, dtype={"patient_id": "string"})
    missing = sorted({"patient_id", "ct_path"} - set(table.columns))
    if missing:
        raise ValueError(f"Sheet {sheet_name!r} is missing required columns: {missing}")
    table = table.loc[:, ["patient_id", "ct_path"]].copy()
    table["patient_id"] = table["patient_id"].map(clean_patient_id)
    table["ct_path"] = table["ct_path"].astype("string").str.strip()
    if table["ct_path"].isna().any() or table["ct_path"].isin([""]).any():
        raise ValueError("Input contains an empty ct_path")
    return table


def one_row_per_patient(table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    rows: list[pd.Series] = []
    conflicts: dict[str, str] = {}
    for patient_id, group in table.groupby("patient_id", sort=False):
        paths = group["ct_path"].drop_duplicates().tolist()
        if len(paths) > 1:
            conflicts[str(patient_id)] = (
                "multiple distinct ct_path values: " + " | ".join(paths)
            )
        rows.append(group.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True), conflicts


def initialize_manifest(table: pd.DataFrame, manifest_path: Path) -> pd.DataFrame:
    manifest = (
        load_manifest(manifest_path)
        if manifest_path.is_file()
        else canonicalize_manifest(pd.DataFrame({"patient_id": table["patient_id"]}))
    )
    manifest = manifest.set_index("patient_id", drop=False)
    for row in table.itertuples(index=False):
        if row.patient_id not in manifest.index:
            manifest.loc[row.patient_id, :] = ""
            manifest.loc[row.patient_id, "patient_id"] = row.patient_id
        manifest.loc[row.patient_id, "source_ct_path"] = row.ct_path
    return canonicalize_manifest(manifest.reset_index(drop=True))


def inspect_image(image: sitk.Image) -> dict[str, float | int]:
    if image.GetDimension() != 3:
        raise ValueError(f"Expected a 3D CT, found {image.GetDimension()}D")
    if not np.isfinite(sitk.GetArrayViewFromImage(image)).all():
        raise ValueError("CT contains non-finite voxel values")
    size, spacing = image.GetSize(), image.GetSpacing()
    if any(value <= 0 or not np.isfinite(value) for value in spacing):
        raise ValueError(f"Invalid CT spacing: {spacing}")
    return {
        "full_ct_shape_x": int(size[0]),
        "full_ct_shape_y": int(size[1]),
        "full_ct_shape_z": int(size[2]),
        "full_ct_spacing_x": float(spacing[0]),
        "full_ct_spacing_y": float(spacing[1]),
        "full_ct_spacing_z": float(spacing[2]),
    }


def read_source(source: Path, min_slices: int) -> tuple[sitk.Image, dict[str, object]]:
    if source.is_file() and source.name.lower().endswith((".nii", ".nii.gz")):
        return sitk.ReadImage(str(source)), {
            "selected_series_uid": "",
            "series_description": "source NIfTI",
            "candidate_series_count": 0,
        }
    if source.is_dir():
        selected, candidates = select_series(source, min_slices)
        return read_series(selected), {
            "selected_series_uid": selected.uid,
            "series_description": selected.description,
            "candidate_series_count": len(candidates),
            "selected_series_image_type": selected.image_type,
            "selected_series_files": len(selected.files),
            "selected_series_coverage_mm": selected.coverage_mm,
        }
    raise ValueError("ct_path must be a DICOM directory or a .nii/.nii.gz file")


def write_full_ct(image: sitk.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name("full_ct.partial.nii.gz")
    sitk.WriteImage(
        orient_lps(sitk.Cast(image, sitk.sitkFloat32)), str(temporary), True
    )
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--sheet", default="baseline")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--patient-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-slices", type=int, default=16)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--relative-paths", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or args.output_root / "ct_manifest.csv"
    all_patients, conflicts = one_row_per_patient(
        load_input_table(args.input, args.sheet)
    )
    manifest = initialize_manifest(all_patients, manifest_path)
    table = all_patients
    if args.patient_id is not None:
        patient = clean_patient_id(args.patient_id)
        table = table.loc[table["patient_id"] == patient]
        if table.empty:
            raise ValueError(f"patient_id={patient!r} is not present")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        table = table.head(args.limit)
    row_by_id = {str(row.patient_id): index for index, row in manifest.iterrows()}
    for row in table.itertuples(index=False):
        patient_id = str(row.patient_id)
        index = row_by_id[patient_id]
        destination = patient_directory(args.output_root, patient_id) / "full_ct.nii.gz"
        try:
            if patient_id in conflicts:
                raise ValueError(conflicts[patient_id])
            if args.skip_existing and destination.is_file():
                image = sitk.ReadImage(str(destination))
                selection = {}
                status, message = "skipped_existing", "existing full_ct passed checks"
            else:
                image, selection = read_source(Path(str(row.ct_path)), args.min_slices)
                write_full_ct(image, destination)
                image = sitk.ReadImage(str(destination))
                status, message = (
                    "success",
                    "canonical LPS float32 NIfTI; HU values not windowed",
                )
            for key, value in {**selection, **inspect_image(image)}.items():
                manifest.loc[index, key] = value
            manifest.loc[index, "full_ct_path"] = relative_manifest_path(
                destination, args.output_root, args.relative_paths
            )
            manifest.loc[index, "full_ct_status"] = status
            manifest.loc[index, "full_ct_message"] = message
            print(
                f"[{patient_id}] {status}: size={image.GetSize()}, spacing={image.GetSpacing()}"
            )
        except Exception as error:
            manifest.loc[index, "full_ct_status"] = "failed"
            manifest.loc[index, "full_ct_message"] = str(error)
            print(f"[{patient_id}] failed: {error}")
        finally:
            write_manifest(manifest, manifest_path)
    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()

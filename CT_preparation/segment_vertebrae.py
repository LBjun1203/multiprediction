#!/usr/bin/env python3
"""Run TotalSegmentator for L2/L3/L4 and record numerical QC."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

LABELS = ("vertebrae_L2", "vertebrae_L3", "vertebrae_L4")
SEGMENTATION_COLUMNS = (
    "L2_mask_path",
    "L3_mask_path",
    "L4_mask_path",
    "L2_voxels",
    "L3_voxels",
    "L4_voxels",
    "segmentation_status",
    "segmentation_message",
)


def clean_patient_id(value: object) -> str:
    if pd.isna(value):
        raise ValueError("patient_id contains a missing value")
    text = str(value).strip()
    if not text:
        raise ValueError("patient_id contains an empty value")
    match = re.fullmatch(r"(\d+)\.0", text)
    return match.group(1) if match else text


def patient_directory(output_root: Path, patient_id: str) -> Path:
    if patient_id in {".", ".."} or Path(patient_id).name != patient_id:
        raise ValueError(f"Unsafe patient_id for an output directory: {patient_id!r}")
    return output_root / patient_id


def resolve_manifest_path(value: object, output_root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else output_root / path


def output_path_value(path: Path, output_root: Path, relative_paths: bool) -> str:
    resolved = path.resolve()
    if not relative_paths:
        return str(resolved)
    try:
        return str(resolved.relative_to(output_root.resolve()))
    except ValueError as error:
        raise ValueError(f"Output path is outside --output-root: {resolved}") from error


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    manifest = pd.read_csv(path, dtype={"patient_id": "string"}, keep_default_na=False)
    required = {"patient_id", "nifti_path", "conversion_status"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    manifest["patient_id"] = manifest["patient_id"].map(clean_patient_id)
    if manifest["patient_id"].duplicated().any():
        duplicate = manifest.loc[
            manifest["patient_id"].duplicated(), "patient_id"
        ].iloc[0]
        raise ValueError(f"Manifest has duplicate patient_id: {duplicate!r}")
    for column in SEGMENTATION_COLUMNS:
        if column not in manifest.columns:
            manifest[column] = pd.Series("", index=manifest.index, dtype="object")
        else:
            manifest[column] = manifest[column].astype("object")
    return manifest


def write_manifest(manifest: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    manifest.to_csv(temporary, index=False)
    temporary.replace(path)


def geometry_matches(reference: sitk.Image, mask: sitk.Image) -> bool:
    return (
        reference.GetDimension() == mask.GetDimension() == 3
        and reference.GetSize() == mask.GetSize()
        and np.allclose(reference.GetSpacing(), mask.GetSpacing(), rtol=0, atol=1e-5)
        and np.allclose(reference.GetOrigin(), mask.GetOrigin(), rtol=0, atol=1e-4)
        and np.allclose(
            reference.GetDirection(), mask.GetDirection(), rtol=0, atol=1e-5
        )
    )


def inspect_masks(ct_path: Path, segmentation_dir: Path) -> dict[str, int]:
    ct = sitk.ReadImage(str(ct_path))
    if ct.GetDimension() != 3:
        raise ValueError(f"Expected a 3D CT, found {ct.GetDimension()}D")
    voxel_counts: dict[str, int] = {}
    for label in LABELS:
        mask_path = segmentation_dir / f"{label}.nii.gz"
        if not mask_path.is_file():
            raise FileNotFoundError(f"missing mask: {mask_path.name}")
        mask = sitk.ReadImage(str(mask_path))
        if not geometry_matches(ct, mask):
            raise ValueError(f"{mask_path.name} geometry is not aligned with CT")
        voxel_count = int(np.count_nonzero(sitk.GetArrayViewFromImage(mask)))
        if voxel_count <= 0:
            raise ValueError(f"{mask_path.name} is empty")
        voxel_counts[label] = voxel_count
    return voxel_counts


def select_indices(
    manifest: pd.DataFrame, patient_id: str | None, limit: int | None
) -> list[int]:
    indices = manifest.index.tolist()
    if patient_id is not None:
        cleaned = clean_patient_id(patient_id)
        matches = manifest.index[manifest["patient_id"] == cleaned].tolist()
        if not matches:
            raise ValueError(f"patient_id={cleaned!r} is not present in the manifest")
        indices = matches
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        indices = indices[:limit]
    return indices


def inspect_total_segmentator_cli(executable: str) -> str:
    version_result = subprocess.run(
        [executable, "--version"], capture_output=True, text=True, check=False
    )
    version_output = (version_result.stdout or version_result.stderr).strip()
    help_result = subprocess.run(
        [executable, "--help"], capture_output=True, text=True, check=False
    )
    help_output = (help_result.stdout or help_result.stderr).strip()
    if help_result.returncode != 0 or "--roi_subset" not in help_output:
        raise RuntimeError(
            "Installed TotalSegmentator CLI does not expose the required --roi_subset option"
        )
    return version_output.splitlines()[0] if version_output else "version unavailable"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--patient-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Store generated paths relative to --output-root for portability",
    )
    args = parser.parse_args()

    executable = shutil.which("totalsegmentator") or shutil.which("TotalSegmentator")
    if executable is None:
        raise FileNotFoundError(
            "totalsegmentator/TotalSegmentator executable was not found on PATH"
        )
    print(f"TotalSegmentator: {inspect_total_segmentator_cli(executable)}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)
    for index in select_indices(manifest, args.patient_id, args.limit):
        patient_id = str(manifest.loc[index, "patient_id"])
        print(f"[{patient_id}] segmentation")
        try:
            if manifest.loc[index, "conversion_status"] not in {
                "success",
                "skipped_existing",
            }:
                manifest.loc[index, "segmentation_status"] = "skipped"
                manifest.loc[index, "segmentation_message"] = (
                    "conversion did not succeed"
                )
                print(f"[{patient_id}] skipped: conversion did not succeed")
                continue
            ct_path = resolve_manifest_path(
                manifest.loc[index, "nifti_path"], args.output_root
            )
            if not ct_path.is_file():
                raise FileNotFoundError(f"NIfTI does not exist: {ct_path}")
            segmentation_dir = (
                patient_directory(args.output_root, patient_id) / "segmentation"
            )
            segmentation_dir.mkdir(parents=True, exist_ok=True)

            expected = [segmentation_dir / f"{label}.nii.gz" for label in LABELS]
            if args.skip_existing and all(path.is_file() for path in expected):
                counts = inspect_masks(ct_path, segmentation_dir)
                status = "skipped_existing"
                message = "existing L2/L3/L4 masks passed numerical QC"
            else:
                command = [
                    executable,
                    "-i",
                    str(ct_path),
                    "-o",
                    str(segmentation_dir),
                    "--roi_subset",
                    *LABELS,
                ]
                result = subprocess.run(
                    command, capture_output=True, text=True, check=False
                )
                if result.returncode != 0:
                    detail = (result.stderr or result.stdout).strip()
                    raise RuntimeError(
                        f"TotalSegmentator failed with exit code {result.returncode}: {detail}"
                    )
                counts = inspect_masks(ct_path, segmentation_dir)
                status = "success"
                message = " ".join(command)

            for label in LABELS:
                short = label.removeprefix("vertebrae_")
                manifest.loc[index, f"{short}_mask_path"] = output_path_value(
                    segmentation_dir / f"{label}.nii.gz",
                    args.output_root,
                    args.relative_paths,
                )
                manifest.loc[index, f"{short}_voxels"] = counts[label]
            manifest.loc[index, "segmentation_status"] = status
            manifest.loc[index, "segmentation_message"] = message
            print(
                f"[{patient_id}] {status}: L2={counts['vertebrae_L2']}, "
                f"L3={counts['vertebrae_L3']}, L4={counts['vertebrae_L4']} voxels"
            )
        except Exception as error:  # noqa: BLE001 - isolate one bad patient
            manifest.loc[index, "segmentation_status"] = "failed"
            manifest.loc[index, "segmentation_message"] = str(error)
            print(f"[{patient_id}] failed: {error}")
        finally:
            write_manifest(manifest, args.manifest)

    print(f"Manifest: {args.manifest.resolve()}")


if __name__ == "__main__":
    main()

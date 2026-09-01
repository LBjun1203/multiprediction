#!/usr/bin/env python3
"""Convert one CT source per patient to a standard ct.nii.gz file."""

from __future__ import annotations

import argparse
import gzip
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

REQUIRED_COLUMNS = ("patient_id", "ct_path")
CONVERSION_COLUMNS = (
    "patient_id",
    "source_ct_path",
    "nifti_path",
    "conversion_status",
    "conversion_message",
    "size_x",
    "size_y",
    "size_z",
    "spacing_x",
    "spacing_y",
    "spacing_z",
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


def output_path_value(path: Path, output_root: Path, relative_paths: bool) -> str:
    resolved = path.resolve()
    if not relative_paths:
        return str(resolved)
    try:
        return str(resolved.relative_to(output_root.resolve()))
    except ValueError as error:
        raise ValueError(f"Output path is outside --output-root: {resolved}") from error


def load_input_table(path: Path, sheet_name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input Excel file does not exist: {path}")
    table = pd.read_excel(path, sheet_name=sheet_name, dtype={"patient_id": "string"})
    missing = sorted(set(REQUIRED_COLUMNS) - set(table.columns))
    if missing:
        raise ValueError(f"Sheet {sheet_name!r} is missing required columns: {missing}")

    table = table.loc[:, REQUIRED_COLUMNS].copy()
    table["patient_id"] = table["patient_id"].map(clean_patient_id)
    table["ct_path"] = table["ct_path"].astype("string").str.strip()
    invalid_path = table["ct_path"].isna() | table["ct_path"].str.lower().isin(
        {"", "nan", "none", "<na>"}
    )
    if invalid_path.any():
        patient_id = table.loc[invalid_path, "patient_id"].iloc[0]
        raise ValueError(f"patient_id={patient_id!r} has an empty ct_path")
    return table


def one_row_per_patient(table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    conflicts: dict[str, str] = {}
    rows: list[pd.Series] = []
    for patient_id, group in table.groupby("patient_id", sort=False):
        paths = group["ct_path"].drop_duplicates().tolist()
        if len(paths) > 1:
            conflicts[patient_id] = "multiple distinct ct_path values: " + " | ".join(
                paths
            )
        rows.append(group.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True), conflicts


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=CONVERSION_COLUMNS)
    manifest = pd.read_csv(path, dtype={"patient_id": "string"}, keep_default_na=False)
    if "patient_id" not in manifest.columns:
        raise ValueError("Manifest is missing patient_id")
    manifest["patient_id"] = manifest["patient_id"].map(clean_patient_id)
    if manifest["patient_id"].duplicated().any():
        duplicate = manifest.loc[
            manifest["patient_id"].duplicated(), "patient_id"
        ].iloc[0]
        raise ValueError(f"Manifest has duplicate patient_id: {duplicate!r}")
    return manifest


def initialize_manifest(input_table: pd.DataFrame, manifest_path: Path) -> pd.DataFrame:
    manifest = load_manifest(manifest_path)
    for column in CONVERSION_COLUMNS:
        if column not in manifest.columns:
            manifest[column] = pd.Series("", index=manifest.index, dtype="object")
        elif column != "patient_id":
            manifest[column] = manifest[column].astype("object")
    manifest = manifest.set_index("patient_id", drop=False)
    for row in input_table.itertuples(index=False):
        if row.patient_id not in manifest.index:
            manifest.loc[row.patient_id, :] = ""
            manifest.loc[row.patient_id, "patient_id"] = row.patient_id
        manifest.loc[row.patient_id, "source_ct_path"] = row.ct_path
    return manifest.reset_index(drop=True)


def write_manifest(manifest: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [c for c in CONVERSION_COLUMNS if c in manifest.columns]
    ordered.extend(c for c in manifest.columns if c not in ordered)
    temporary = path.with_name(path.name + ".tmp")
    manifest.loc[:, ordered].to_csv(temporary, index=False)
    temporary.replace(path)


def inspect_nifti(path: Path) -> dict[str, float | int]:
    image = sitk.ReadImage(str(path))
    if image.GetDimension() != 3:
        raise ValueError(f"Expected a 3D CT, found {image.GetDimension()}D")
    size = image.GetSize()
    spacing = image.GetSpacing()
    if any(value <= 0 or not np.isfinite(value) for value in spacing):
        raise ValueError(f"Invalid spacing: {spacing}")
    if len(image.GetDirection()) != 9 or not np.isfinite(image.GetDirection()).all():
        raise ValueError("Image direction contains non-finite values")
    if not np.isfinite(image.GetOrigin()).all():
        raise ValueError("Image origin contains non-finite values")
    array = sitk.GetArrayViewFromImage(image)
    if not np.isfinite(array).all():
        raise ValueError("CT contains non-finite voxel values")
    return {
        "size_x": int(size[0]),
        "size_y": int(size[1]),
        "size_z": int(size[2]),
        "spacing_x": float(spacing[0]),
        "spacing_y": float(spacing[1]),
        "spacing_z": float(spacing[2]),
    }


def copy_nifti(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    temporary = destination.with_name(
        destination.name.removesuffix(".nii.gz") + ".partial.nii.gz"
    )
    if temporary.exists():
        temporary.unlink()
    if source.name.lower().endswith(".nii.gz"):
        shutil.copy2(source, temporary)
    else:
        with (
            source.open("rb") as source_file,
            gzip.open(temporary, "wb") as output_file,
        ):
            shutil.copyfileobj(source_file, output_file)
    inspect_nifti(temporary)
    temporary.replace(destination)


def convert_dicom_directory(source: Path, destination: Path, executable: str) -> str:
    with tempfile.TemporaryDirectory(
        prefix="dcm2niix_", dir=destination.parent
    ) as temp_name:
        temp_dir = Path(temp_name)
        command = [
            executable,
            "-z",
            "y",
            "-f",
            "ct",
            "-o",
            str(temp_dir),
            str(source),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(
                f"dcm2niix failed with exit code {result.returncode}: {detail}"
            )
        outputs = sorted(
            path
            for path in temp_dir.iterdir()
            if path.is_file() and path.name.lower().endswith((".nii", ".nii.gz"))
        )
        if len(outputs) != 1:
            names = ", ".join(path.name for path in outputs) or "none"
            raise RuntimeError(
                f"multiple_nifti_outputs: expected 1, found {len(outputs)} ({names})"
            )
        copy_nifti(outputs[0], destination)
        return " ".join(command)


def selected_patient_ids(
    input_table: pd.DataFrame, patient_id: str | None, limit: int | None
) -> list[str]:
    ids = input_table["patient_id"].tolist()
    if patient_id is not None:
        cleaned = clean_patient_id(patient_id)
        if cleaned not in ids:
            raise ValueError(
                f"patient_id={cleaned!r} is not present in the selected sheet"
            )
        ids = [cleaned]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        ids = ids[:limit]
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="Cleaned Huashan .xlsx file"
    )
    parser.add_argument("--sheet", default="baseline")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--patient-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Store generated paths relative to --output-root for portability",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    if not args.output_root.is_dir():
        raise NotADirectoryError(f"Cannot create output root: {args.output_root}")
    manifest_path = args.manifest or args.output_root / "ct_manifest.csv"

    raw_table = load_input_table(args.input, args.sheet)
    input_table, conflicts = one_row_per_patient(raw_table)
    manifest = initialize_manifest(input_table, manifest_path)
    selected_ids = selected_patient_ids(input_table, args.patient_id, args.limit)
    source_by_id = input_table.set_index("patient_id")["ct_path"].to_dict()

    needs_dcm2niix = any(
        Path(source_by_id[patient_id]).is_dir()
        for patient_id in selected_ids
        if patient_id not in conflicts and Path(source_by_id[patient_id]).exists()
    )
    dcm2niix = shutil.which("dcm2niix")
    if needs_dcm2niix and dcm2niix is None:
        raise FileNotFoundError("dcm2niix executable was not found on PATH")

    row_by_id = {str(row.patient_id): index for index, row in manifest.iterrows()}
    for patient_id in selected_ids:
        index = row_by_id[patient_id]
        source = Path(source_by_id[patient_id])
        destination_dir = patient_directory(args.output_root, patient_id)
        destination = destination_dir / "ct.nii.gz"
        print(f"[{patient_id}] source={source}")
        try:
            if patient_id in conflicts:
                raise ValueError(conflicts[patient_id])
            if not source.exists():
                raise FileNotFoundError("source path not found")
            destination_dir.mkdir(parents=True, exist_ok=True)

            if args.skip_existing and destination.is_file():
                metadata = inspect_nifti(destination)
                status = "skipped_existing"
                message = "existing ct.nii.gz passed basic NIfTI checks"
            else:
                is_nifti = source.is_file() and source.name.lower().endswith(
                    (".nii", ".nii.gz")
                )
                if is_nifti:
                    copy_nifti(source, destination)
                    command_text = "existing NIfTI copied/gzipped; dcm2niix not used"
                elif source.is_dir():
                    if dcm2niix is None:
                        raise FileNotFoundError(
                            "dcm2niix executable was not found on PATH"
                        )
                    command_text = convert_dicom_directory(
                        source, destination, dcm2niix
                    )
                else:
                    raise ValueError(
                        "ct_path must be a DICOM directory or a .nii/.nii.gz file"
                    )
                metadata = inspect_nifti(destination)
                status = "success"
                message = command_text

            manifest.loc[index, "nifti_path"] = output_path_value(
                destination, args.output_root, args.relative_paths
            )
            manifest.loc[index, "conversion_status"] = status
            manifest.loc[index, "conversion_message"] = message
            for key, value in metadata.items():
                manifest.loc[index, key] = value
            print(
                f"[{patient_id}] {status}: shape=({metadata['size_x']}, {metadata['size_y']}, "
                f"{metadata['size_z']}), spacing=({metadata['spacing_x']}, "
                f"{metadata['spacing_y']}, {metadata['spacing_z']})"
            )
        except Exception as error:  # noqa: BLE001 - isolate one bad patient
            manifest.loc[index, "conversion_status"] = "failed"
            manifest.loc[index, "conversion_message"] = str(error)
            print(f"[{patient_id}] failed: {error}")
        finally:
            write_manifest(manifest, manifest_path)

    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()

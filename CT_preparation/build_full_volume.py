#!/usr/bin/env python3
"""Build one full-coverage windowed CT ``.npy`` volume per patient.

The input layout is expected to be ``<input-root>/<patient_id>/*.dcm``.  The
script selects one diagnostic axial CT series, preserves its complete
inferior-superior coverage, canonicalizes it to LPS, resamples it, applies an
HU window, and writes ``[D,H,W]`` float32 arrays in ``[0,1]``.

This is a separate preprocessing branch from ``extract_l2_l4_roi.py`` and
``build_25d.py``.  It never crops the scan to the lumbar spine.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk


DEFAULT_INPUT_ROOT = Path(r"E:\data\huashan_349")
DEFAULT_OUTPUT_ROOT = Path(r"E:\data\huashan_349_processed")
LOCALIZER_PATTERN = re.compile(
    r"locali[sz]er|scout|topogram|surview|scanogram|定位|定位像", re.IGNORECASE
)
MANIFEST_COLUMNS = (
    "patient_id",
    "source_dir",
    "candidate_series_count",
    "selected_series_uid",
    "series_description",
    "image_type",
    "num_dicom_files",
    "coverage_mm",
    "original_shape_x",
    "original_shape_y",
    "original_shape_z",
    "original_spacing_x",
    "original_spacing_y",
    "original_spacing_z",
    "resampled_shape_x",
    "resampled_shape_y",
    "resampled_shape_z",
    "resampled_spacing_x",
    "resampled_spacing_y",
    "resampled_spacing_z",
    "npy_full_path",
    "npy_shape_z",
    "npy_shape_y",
    "npy_shape_x",
    "image_size",
    "window_min",
    "window_max",
    "dtype",
    "min_value",
    "max_value",
    "build_full_status",
    "build_full_message",
)


@dataclass(frozen=True)
class SeriesCandidate:
    uid: str
    files: tuple[str, ...]
    description: str
    image_type: str
    coverage_mm: float

    @property
    def score(self) -> tuple[float, int, str]:
        # Anatomical coverage is more important than slice thickness/count.
        return (self.coverage_mm, len(self.files), self.uid)


def clean_patient_id(value: object) -> str:
    if pd.isna(value):
        raise ValueError("patient_id contains a missing value")
    text = str(value).strip()
    if not text:
        raise ValueError("patient_id contains an empty value")
    match = re.fullmatch(r"(\d+)\.0", text)
    return match.group(1) if match else text


def patient_directory(root: Path, patient_id: str) -> Path:
    if patient_id in {".", ".."} or Path(patient_id).name != patient_id:
        raise ValueError(f"Unsafe patient_id for a directory: {patient_id!r}")
    return root / patient_id


def output_path_value(path: Path, output_root: Path, relative_paths: bool) -> str:
    resolved = path.resolve()
    if not relative_paths:
        return str(resolved)
    try:
        return str(resolved.relative_to(output_root.resolve()))
    except ValueError as error:
        raise ValueError(f"Output path is outside --output-root: {resolved}") from error


def metadata_value(reader: sitk.ImageFileReader, key: str) -> str:
    return reader.GetMetaData(key).strip() if reader.HasMetaDataKey(key) else ""


def read_dicom_metadata(path: str) -> dict[str, str]:
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    return {
        "modality": metadata_value(reader, "0008|0060"),
        "description": metadata_value(reader, "0008|103e"),
        "image_type": metadata_value(reader, "0008|0008"),
        "position": metadata_value(reader, "0020|0032"),
        "orientation": metadata_value(reader, "0020|0037"),
    }


def parse_dicom_vector(text: str, expected_length: int) -> np.ndarray | None:
    try:
        values = np.asarray([float(value) for value in text.split("\\")], dtype=float)
    except ValueError:
        return None
    if values.shape != (expected_length,) or not np.isfinite(values).all():
        return None
    return values


def series_geometry(first: dict[str, str], last: dict[str, str]) -> tuple[bool, float]:
    orientation = parse_dicom_vector(first["orientation"], 6)
    first_position = parse_dicom_vector(first["position"], 3)
    last_position = parse_dicom_vector(last["position"], 3)
    if orientation is None:
        return False, 0.0
    normal = np.cross(orientation[:3], orientation[3:])
    norm = float(np.linalg.norm(normal))
    if norm <= 0 or not math.isfinite(norm):
        return False, 0.0
    normal /= norm
    is_axial = abs(float(normal[2])) >= 0.8
    if first_position is None or last_position is None:
        return is_axial, 0.0
    coverage = abs(float(np.dot(last_position - first_position, normal)))
    return is_axial, coverage


def discover_series(patient_dir: Path, min_slices: int) -> list[SeriesCandidate]:
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(patient_dir)) or []
    candidates: list[SeriesCandidate] = []
    for uid in series_ids:
        files = tuple(
            sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(patient_dir), uid)
        )
        if len(files) < min_slices:
            continue
        first = read_dicom_metadata(files[0])
        if first["modality"].upper() != "CT":
            continue
        searchable = f"{first['description']} {first['image_type']}"
        if LOCALIZER_PATTERN.search(searchable):
            continue
        last = read_dicom_metadata(files[-1])
        is_axial, coverage_mm = series_geometry(first, last)
        if not is_axial:
            continue
        candidates.append(
            SeriesCandidate(
                uid=str(uid),
                files=files,
                description=first["description"],
                image_type=first["image_type"],
                coverage_mm=coverage_mm,
            )
        )
    return candidates


def select_series(
    patient_dir: Path, min_slices: int
) -> tuple[SeriesCandidate, list[SeriesCandidate]]:
    candidates = discover_series(patient_dir, min_slices)
    if not candidates:
        raise ValueError(
            "No diagnostic axial CT series remained after modality, localizer, "
            f"orientation, and minimum-slice ({min_slices}) checks"
        )
    selected = max(candidates, key=lambda candidate: candidate.score)
    return selected, candidates


def read_series(candidate: SeriesCandidate) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(candidate.files)
    image = reader.Execute()
    if image.GetDimension() != 3:
        raise ValueError(f"Expected a 3D CT, found {image.GetDimension()}D")
    if any(value <= 0 or not math.isfinite(value) for value in image.GetSpacing()):
        raise ValueError(f"Invalid CT spacing: {image.GetSpacing()}")
    return sitk.Cast(image, sitk.sitkFloat32)


def orient_lps(image: sitk.Image) -> sitk.Image:
    oriented = sitk.DICOMOrient(image, "LPS")
    orientation = sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
        oriented.GetDirection()
    )
    if orientation != "LPS":
        raise RuntimeError(
            f"Orientation normalization failed: expected LPS, found {orientation}"
        )
    return oriented


def resample_ct(
    image: sitk.Image, target_spacing: tuple[float, float, float]
) -> sitk.Image:
    old_size = image.GetSize()
    old_spacing = image.GetSpacing()
    new_size = [
        max(
            1,
            round((old_size[axis] - 1) * old_spacing[axis] / target_spacing[axis])
            + 1,
        )
        for axis in range(3)
    ]
    return sitk.Resample(
        image,
        new_size,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkLinear,
        image.GetOrigin(),
        target_spacing,
        image.GetDirection(),
        -1024.0,
        sitk.sitkFloat32,
    )


def resize_slice(image_array: np.ndarray, image_size: int) -> np.ndarray:
    image = sitk.GetImageFromArray(image_array.astype(np.float32, copy=False))
    old_size = image.GetSize()
    output_spacing = [
        (old_size[axis] - 1) / (image_size - 1) if old_size[axis] > 1 else 1.0
        for axis in range(2)
    ]
    resized = sitk.Resample(
        image,
        [image_size, image_size],
        sitk.Transform(2, sitk.sitkIdentity),
        sitk.sitkLinear,
        image.GetOrigin(),
        output_spacing,
        image.GetDirection(),
        0.0,
        sitk.sitkFloat32,
    )
    return sitk.GetArrayFromImage(resized).astype(np.float32, copy=False)


def build_full_array(
    image: sitk.Image,
    output_path: Path,
    image_size: int,
    window_min: float,
    window_max: float,
) -> tuple[tuple[int, int, int], float, float]:
    volume = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    if volume.ndim != 3:
        raise ValueError(f"Expected [D,H,W] volume, found shape {volume.shape}")
    if not np.isfinite(volume).all():
        raise ValueError("Resampled CT contains non-finite voxel values")

    depth = int(volume.shape[0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name("full_ct.partial.npy")
    if temporary.exists():
        temporary.unlink()
    output = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(depth, image_size, image_size),
    )
    minimum = math.inf
    maximum = -math.inf
    try:
        for index in range(depth):
            axial = np.clip(volume[index], window_min, window_max)
            axial = (axial - window_min) / (window_max - window_min)
            axial = np.clip(resize_slice(axial, image_size), 0.0, 1.0)
            output[index] = axial
            minimum = min(minimum, float(axial.min()))
            maximum = max(maximum, float(axial.max()))
        output.flush()
    except Exception:
        del output
        if temporary.exists():
            temporary.unlink()
        raise
    del output
    temporary.replace(output_path)
    return (depth, image_size, image_size), minimum, maximum


def validate_existing_array(path: Path, image_size: int) -> np.ndarray:
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    if array.ndim != 3 or tuple(array.shape[1:]) != (image_size, image_size):
        raise ValueError(
            f"Existing array shape {array.shape} is not [D,{image_size},{image_size}]"
        )
    if array.shape[0] <= 0:
        raise ValueError("Existing array has zero depth")
    if array.dtype != np.float32:
        raise ValueError(f"Existing array dtype {array.dtype} != float32")
    if not np.isfinite(array).all():
        raise ValueError("Existing array contains non-finite values")
    if float(array.min()) < 0.0 or float(array.max()) > 1.0:
        raise ValueError("Existing array is outside [0,1]")
    return array


def save_preview(array_path: Path, preview_path: Path) -> None:
    array = np.load(array_path, allow_pickle=False, mmap_mode="r")
    indices = np.linspace(0, array.shape[0] - 1, num=min(9, array.shape[0]))
    indices = np.rint(indices).astype(int)
    preview = np.concatenate(
        [np.rint(array[index] * 255.0).astype(np.uint8) for index in indices], axis=1
    )
    sitk.WriteImage(sitk.GetImageFromArray(preview), str(preview_path))


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    manifest = pd.read_csv(path, dtype={"patient_id": "string"}, keep_default_na=False)
    if "patient_id" not in manifest.columns:
        raise ValueError("Full CT manifest is missing patient_id")
    manifest["patient_id"] = manifest["patient_id"].map(clean_patient_id)
    if manifest["patient_id"].duplicated().any():
        duplicate = manifest.loc[
            manifest["patient_id"].duplicated(), "patient_id"
        ].iloc[0]
        raise ValueError(f"Full CT manifest has duplicate patient_id: {duplicate!r}")
    for column in MANIFEST_COLUMNS:
        if column not in manifest.columns:
            manifest[column] = pd.Series("", index=manifest.index, dtype="object")
        elif column != "patient_id":
            manifest[column] = manifest[column].astype("object")
    return manifest


def ensure_manifest_rows(
    manifest: pd.DataFrame, patient_dirs: list[Path]
) -> pd.DataFrame:
    manifest = manifest.set_index("patient_id", drop=False)
    for directory in patient_dirs:
        patient_id = clean_patient_id(directory.name)
        if patient_id not in manifest.index:
            manifest.loc[patient_id, :] = ""
            manifest.loc[patient_id, "patient_id"] = patient_id
        manifest.loc[patient_id, "source_dir"] = str(directory.resolve())
    return manifest.reset_index(drop=True)


def write_manifest(manifest: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [column for column in MANIFEST_COLUMNS if column in manifest.columns]
    ordered.extend(column for column in manifest.columns if column not in ordered)
    temporary = path.with_name(path.name + ".tmp")
    manifest.loc[:, ordered].to_csv(temporary, index=False)
    temporary.replace(path)


def selected_directories(
    input_root: Path, patient_id: str | None, limit: int | None
) -> list[Path]:
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root does not exist: {input_root}")
    directories = sorted(path for path in input_root.iterdir() if path.is_dir())
    if patient_id is not None:
        cleaned = clean_patient_id(patient_id)
        directories = [path for path in directories if path.name == cleaned]
        if not directories:
            raise ValueError(f"patient_id={cleaned!r} is not present in --input-root")
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        directories = directories[:limit]
    if not directories:
        raise ValueError(f"No patient directories were found in {input_root}")
    return directories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--target-spacing",
        type=float,
        nargs=3,
        default=(1.5, 1.5, 2.0),
        metavar=("SX", "SY", "SZ"),
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--window-min", type=float, default=-200.0)
    parser.add_argument("--window-max", type=float, default=300.0)
    parser.add_argument("--min-slices", type=int, default=16)
    parser.add_argument("--patient-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--save-preview", action="store_true")
    parser.add_argument("--relative-paths", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="select and report DICOM series without reading/resampling CT pixels",
    )
    args = parser.parse_args()

    target_spacing = tuple(float(value) for value in args.target_spacing)
    if any(value <= 0 or not math.isfinite(value) for value in target_spacing):
        raise ValueError("--target-spacing values must be finite and positive")
    if args.image_size < 32:
        raise ValueError("--image-size must be at least 32")
    if args.min_slices <= 0:
        raise ValueError("--min-slices must be positive")
    if not np.isfinite([args.window_min, args.window_max]).all():
        raise ValueError("Window bounds must be finite")
    if args.window_min >= args.window_max:
        raise ValueError("--window-min must be smaller than --window-max")

    patient_dirs = selected_directories(args.input_root, args.patient_id, args.limit)
    manifest_path = args.manifest or args.output_root / "full_ct_manifest.csv"
    manifest = ensure_manifest_rows(load_manifest(manifest_path), patient_dirs)
    row_by_id = {str(row.patient_id): index for index, row in manifest.iterrows()}

    for patient_dir in patient_dirs:
        patient_id = clean_patient_id(patient_dir.name)
        index = row_by_id[patient_id]
        print(f"[{patient_id}] full CT")
        try:
            selected, candidates = select_series(patient_dir, args.min_slices)
            manifest.loc[index, "candidate_series_count"] = len(candidates)
            manifest.loc[index, "selected_series_uid"] = selected.uid
            manifest.loc[index, "series_description"] = selected.description
            manifest.loc[index, "image_type"] = selected.image_type
            manifest.loc[index, "num_dicom_files"] = len(selected.files)
            manifest.loc[index, "coverage_mm"] = selected.coverage_mm
            selection_text = (
                f"selected UID={selected.uid}, coverage={selected.coverage_mm:.1f} mm, "
                f"files={len(selected.files)}, candidates={len(candidates)}"
            )

            if args.dry_run:
                manifest.loc[index, "build_full_status"] = "dry_run"
                manifest.loc[index, "build_full_message"] = selection_text
                print(f"[{patient_id}] dry run: {selection_text}")
                continue

            output_dir = patient_directory(args.output_root, patient_id)
            output_path = output_dir / "full_ct.npy"
            if args.skip_existing and output_path.is_file():
                existing = validate_existing_array(output_path, args.image_size)
                npy_shape = tuple(int(value) for value in existing.shape)
                minimum = float(existing.min())
                maximum = float(existing.max())
                status = "skipped_existing"
                message = "existing full CT array passed shape, dtype, finite, and range checks"
                del existing
                original_size = ("", "", "")
                original_spacing = ("", "", "")
                resampled_size = (args.image_size, args.image_size, npy_shape[0])
                resampled_spacing = ("", "", target_spacing[2])
            else:
                image = read_series(selected)
                original_size = tuple(int(value) for value in image.GetSize())
                original_spacing = tuple(float(value) for value in image.GetSpacing())
                resampled = resample_ct(orient_lps(image), target_spacing)
                resampled_size = tuple(int(value) for value in resampled.GetSize())
                resampled_spacing = tuple(float(value) for value in resampled.GetSpacing())
                npy_shape, minimum, maximum = build_full_array(
                    resampled,
                    output_path,
                    args.image_size,
                    args.window_min,
                    args.window_max,
                )
                status = "success"
                message = (
                    selection_text
                    + "; LPS orientation, linear resampling, HU window, and linear resize"
                )

            if args.save_preview:
                save_preview(output_path, output_dir / "full_ct_preview.png")

            values = {
                "source_dir": str(patient_dir.resolve()),
                "original_shape_x": original_size[0],
                "original_shape_y": original_size[1],
                "original_shape_z": original_size[2],
                "original_spacing_x": original_spacing[0],
                "original_spacing_y": original_spacing[1],
                "original_spacing_z": original_spacing[2],
                "resampled_shape_x": resampled_size[0],
                "resampled_shape_y": resampled_size[1],
                "resampled_shape_z": resampled_size[2],
                "resampled_spacing_x": resampled_spacing[0],
                "resampled_spacing_y": resampled_spacing[1],
                "resampled_spacing_z": resampled_spacing[2],
                "npy_full_path": output_path_value(
                    output_path, args.output_root, args.relative_paths
                ),
                "npy_shape_z": npy_shape[0],
                "npy_shape_y": npy_shape[1],
                "npy_shape_x": npy_shape[2],
                "image_size": args.image_size,
                "window_min": args.window_min,
                "window_max": args.window_max,
                "dtype": "float32",
                "min_value": minimum,
                "max_value": maximum,
                "build_full_status": status,
                "build_full_message": message,
            }
            for key, value in values.items():
                manifest.loc[index, key] = value
            print(
                f"[{patient_id}] {status}: shape={npy_shape}, dtype=float32, "
                f"min={minimum:.6f}, max={maximum:.6f}"
            )
        except Exception as error:  # noqa: BLE001 - isolate one bad patient
            manifest.loc[index, "build_full_status"] = "failed"
            manifest.loc[index, "build_full_message"] = str(error)
            print(f"[{patient_id}] failed: {error}")
        finally:
            if not args.dry_run:
                write_manifest(manifest, manifest_path)

    if args.dry_run:
        print("Dry run complete; no files were written.")
    else:
        print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()

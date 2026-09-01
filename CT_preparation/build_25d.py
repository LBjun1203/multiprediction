#!/usr/bin/env python3
"""Build a windowed [C,H,W] axial slice stack centered at L3."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

BUILD_COLUMNS = (
    "input_roi_path",
    "npy_path",
    "npy_25d_path",
    "num_slices",
    "center_slice",
    "image_size",
    "window_min",
    "window_max",
    "min_value",
    "max_value",
    "build_25d_status",
    "build_25d_message",
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
    required = {"patient_id", "roi_path", "L3_center_slice", "roi_status"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    manifest["patient_id"] = manifest["patient_id"].map(clean_patient_id)
    if manifest["patient_id"].duplicated().any():
        duplicate = manifest.loc[
            manifest["patient_id"].duplicated(), "patient_id"
        ].iloc[0]
        raise ValueError(f"Manifest has duplicate patient_id: {duplicate!r}")
    for column in BUILD_COLUMNS:
        if column not in manifest.columns:
            manifest[column] = pd.Series("", index=manifest.index, dtype="object")
        else:
            manifest[column] = manifest[column].astype("object")
    return manifest


def write_manifest(manifest: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    manifest.to_csv(temporary, index=False)
    temporary.replace(path)


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


def build_stack(
    roi_path: Path,
    center_slice: int,
    num_slices: int,
    image_size: int,
    window_min: float,
    window_max: float,
) -> np.ndarray:
    if not roi_path.is_file():
        raise FileNotFoundError(f"ROI does not exist: {roi_path}")
    roi = sitk.ReadImage(str(roi_path))
    if roi.GetDimension() != 3:
        raise ValueError(f"Expected a 3D ROI, found {roi.GetDimension()}D")
    orientation = sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
        roi.GetDirection()
    )
    if orientation != "LPS":
        raise ValueError(f"ROI must be LPS-oriented, found {orientation}")

    # extract_l2_l4_roi.py canonicalizes to LPS. SimpleITK returns arrays as
    # [z,y,x], so NumPy axis 0 is now explicitly the axial slice axis.
    volume = sitk.GetArrayFromImage(roi).astype(np.float32, copy=False)
    if not np.isfinite(volume).all():
        raise ValueError("ROI contains non-finite voxel values")
    depth = volume.shape[0]
    if not (0 <= center_slice < depth):
        raise ValueError(f"L3 center slice {center_slice} is outside ROI depth {depth}")

    half = num_slices // 2
    indices = np.clip(
        np.arange(center_slice - half, center_slice + half + 1), 0, depth - 1
    )
    channels: list[np.ndarray] = []
    for slice_index in indices:
        axial = np.clip(volume[int(slice_index)], window_min, window_max)
        axial = (axial - window_min) / (window_max - window_min)
        channels.append(resize_slice(axial, image_size))
    stack = np.stack(channels, axis=0).astype(np.float32, copy=False)
    stack = np.clip(stack, 0.0, 1.0).astype(np.float32, copy=False)
    if stack.shape != (num_slices, image_size, image_size):
        raise RuntimeError(f"Unexpected output shape: {stack.shape}")
    if not np.isfinite(stack).all():
        raise ValueError("2.5D stack contains non-finite values")
    return stack


def save_stack(stack: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("l3_25d.partial.npy")
    np.save(temporary, stack, allow_pickle=False)
    temporary.replace(path)


def save_preview(stack: np.ndarray, path: Path) -> None:
    center_channel = stack[stack.shape[0] // 2]
    preview = np.rint(center_channel * 255.0).astype(np.uint8)
    sitk.WriteImage(sitk.GetImageFromArray(preview), str(path))


def validate_existing_stack(
    path: Path, expected_shape: tuple[int, int, int]
) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    if array.shape != expected_shape:
        raise ValueError(
            f"Existing array shape {array.shape} != expected {expected_shape}"
        )
    if array.dtype != np.float32:
        raise ValueError(f"Existing array dtype {array.dtype} != float32")
    if not np.isfinite(array).all():
        raise ValueError("Existing array contains non-finite values")
    if float(array.min()) < 0.0 or float(array.max()) > 1.0:
        raise ValueError("Existing array is outside [0, 1]")
    return array


def select_indices(
    manifest: pd.DataFrame, patient_id: str | None, limit: int | None
) -> list[int]:
    indices = manifest.index.tolist()
    if patient_id is not None:
        cleaned = clean_patient_id(patient_id)
        indices = manifest.index[manifest["patient_id"] == cleaned].tolist()
        if not indices:
            raise ValueError(f"patient_id={cleaned!r} is not present in the manifest")
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        indices = indices[:limit]
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--num-slices", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--window-min", type=float, default=-200.0)
    parser.add_argument("--window-max", type=float, default=300.0)
    parser.add_argument("--patient-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--save-preview", action="store_true")
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Store generated paths relative to --output-root for portability",
    )
    args = parser.parse_args()

    if args.num_slices <= 0 or args.num_slices % 2 == 0:
        raise ValueError("--num-slices must be a positive odd integer")
    if args.image_size < 2:
        raise ValueError("--image-size must be at least 2")
    if not np.isfinite([args.window_min, args.window_max]).all():
        raise ValueError("Window bounds must be finite")
    if args.window_min >= args.window_max:
        raise ValueError("--window-min must be smaller than --window-max")

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)
    expected_shape = (args.num_slices, args.image_size, args.image_size)

    for index in select_indices(manifest, args.patient_id, args.limit):
        patient_id = str(manifest.loc[index, "patient_id"])
        print(f"[{patient_id}] 2.5D")
        try:
            if manifest.loc[index, "roi_status"] not in {"success", "skipped_existing"}:
                manifest.loc[index, "build_25d_status"] = "skipped"
                manifest.loc[index, "build_25d_message"] = "ROI did not succeed"
                print(f"[{patient_id}] skipped: ROI did not succeed")
                continue
            roi_path = resolve_manifest_path(
                manifest.loc[index, "roi_path"], args.output_root
            )
            center_slice = int(float(manifest.loc[index, "L3_center_slice"]))
            output_dir = patient_directory(args.output_root, patient_id)
            output_path = output_dir / "l3_25d.npy"

            if args.skip_existing and output_path.is_file():
                stack = validate_existing_stack(output_path, expected_shape)
                status = "skipped_existing"
                message = (
                    "existing 2.5D array passed shape, dtype, finite, and range checks"
                )
            else:
                stack = build_stack(
                    roi_path,
                    center_slice,
                    args.num_slices,
                    args.image_size,
                    args.window_min,
                    args.window_max,
                )
                save_stack(stack, output_path)
                status = "success"
                message = "adjacent axial slices with boundary replication, HU window, and linear resize"
            if args.save_preview:
                save_preview(stack, output_dir / "l3_preview.png")

            values = {
                "input_roi_path": output_path_value(
                    roi_path, args.output_root, args.relative_paths
                ),
                "npy_path": output_path_value(
                    output_path, args.output_root, args.relative_paths
                ),
                "npy_25d_path": output_path_value(
                    output_path, args.output_root, args.relative_paths
                ),
                "num_slices": args.num_slices,
                "center_slice": center_slice,
                "image_size": args.image_size,
                "window_min": args.window_min,
                "window_max": args.window_max,
                "min_value": float(stack.min()),
                "max_value": float(stack.max()),
                "build_25d_status": status,
                "build_25d_message": message,
            }
            for key, value in values.items():
                manifest.loc[index, key] = value
            print(
                f"[{patient_id}] {status}: shape={stack.shape}, dtype={stack.dtype}, "
                f"min={stack.min():.6f}, max={stack.max():.6f}"
            )
        except Exception as error:  # noqa: BLE001 - isolate one bad patient
            manifest.loc[index, "build_25d_status"] = "failed"
            manifest.loc[index, "build_25d_message"] = str(error)
            print(f"[{patient_id}] failed: {error}")
        finally:
            write_manifest(manifest, args.manifest)

    print(f"Manifest: {args.manifest.resolve()}")


if __name__ == "__main__":
    main()

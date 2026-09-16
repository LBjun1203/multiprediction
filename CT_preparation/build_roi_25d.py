#!/usr/bin/env python3
"""Build the baseline-equivalent L3-centered ``roi_25d`` representation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from ct_preparation.common import (
    clean_patient_id,
    patient_directory,
    relative_manifest_path,
    resolve_manifest_path,
)
from ct_preparation.manifest import SUCCESS_STATUSES, load_manifest, write_manifest

PREPROCESSING_VERSION = "roi25d-v1"


def resize_slice(image_array: np.ndarray, image_size: int) -> np.ndarray:
    """Baseline interpolation retained byte-for-byte in numerical behavior."""
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
        raise FileNotFoundError(f"L2-L4 ROI does not exist: {roi_path}")
    roi = sitk.ReadImage(str(roi_path))
    if roi.GetDimension() != 3:
        raise ValueError(f"Expected a 3D ROI, found {roi.GetDimension()}D")
    orientation = sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
        roi.GetDirection()
    )
    if orientation != "LPS":
        raise ValueError(f"ROI must be LPS-oriented, found {orientation}")
    volume = sitk.GetArrayFromImage(roi).astype(np.float32, copy=False)
    if not np.isfinite(volume).all():
        raise ValueError("ROI contains non-finite voxel values")
    depth = volume.shape[0]
    if not 0 <= center_slice < depth:
        raise ValueError(f"L3 center slice {center_slice} is outside ROI depth {depth}")
    half = num_slices // 2
    indices = np.clip(
        np.arange(center_slice - half, center_slice + half + 1), 0, depth - 1
    )
    channels = []
    for slice_index in indices:
        axial = np.clip(volume[int(slice_index)], window_min, window_max)
        axial = (axial - window_min) / (window_max - window_min)
        channels.append(resize_slice(axial, image_size))
    stack = np.stack(channels, axis=0).astype(np.float32, copy=False)
    stack = np.clip(stack, 0.0, 1.0).astype(np.float32, copy=False)
    if (
        stack.shape != (num_slices, image_size, image_size)
        or not np.isfinite(stack).all()
    ):
        raise ValueError(f"Invalid roi_25d output shape/content: {stack.shape}")
    return stack


def save_stack(stack: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("roi_25d.partial.npy")
    np.save(temporary, stack, allow_pickle=False)
    temporary.replace(path)


def save_preview(stack: np.ndarray, path: Path) -> None:
    preview = np.rint(stack[stack.shape[0] // 2] * 255.0).astype(np.uint8)
    sitk.WriteImage(sitk.GetImageFromArray(preview), str(path))


def _indices(manifest, patient_id: str | None, limit: int | None) -> list[int]:
    indices = manifest.index.tolist()
    if patient_id is not None:
        cleaned = clean_patient_id(patient_id)
        indices = manifest.index[manifest["patient_id"] == cleaned].tolist()
        if not indices:
            raise ValueError(f"patient_id={cleaned!r} is not present")
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
    parser.add_argument("--relative-paths", action="store_true")
    args = parser.parse_args()
    if args.num_slices <= 0 or args.num_slices % 2 == 0:
        raise ValueError("--num-slices must be a positive odd integer")
    if args.image_size < 2 or args.window_min >= args.window_max:
        raise ValueError("Invalid image size or HU window")
    manifest = load_manifest(args.manifest)
    for index in _indices(manifest, args.patient_id, args.limit):
        patient_id = str(manifest.loc[index, "patient_id"])
        try:
            if (
                str(manifest.loc[index, "l2_l4_roi_status"]).lower()
                not in SUCCESS_STATUSES
            ):
                raise ValueError("l2_l4_roi did not succeed")
            roi_path = resolve_manifest_path(
                manifest.loc[index, "l2_l4_roi_path"], args.output_root
            )
            center = int(float(manifest.loc[index, "l3_center_index"]))
            output = patient_directory(args.output_root, patient_id) / "roi_25d.npy"
            preview = (
                patient_directory(args.output_root, patient_id) / "roi_25d_preview.png"
            )
            if args.skip_existing and output.is_file():
                stack = np.load(output, allow_pickle=False)
                if (
                    stack.shape != (args.num_slices, args.image_size, args.image_size)
                    or stack.dtype != np.float32
                ):
                    raise ValueError(
                        "Existing roi_25d does not match requested contract"
                    )
                status = "skipped_existing"
            else:
                stack = build_stack(
                    roi_path,
                    center,
                    args.num_slices,
                    args.image_size,
                    args.window_min,
                    args.window_max,
                )
                save_stack(stack, output)
                save_preview(stack, preview)
                status = "success"
            values = {
                "roi_25d_path": relative_manifest_path(
                    output, args.output_root, args.relative_paths
                ),
                "roi_25d_status": status,
                "roi_25d_message": "adjacent axial slices; boundary replication; HU window; baseline linear resize",
                "roi_25d_shape_c": stack.shape[0],
                "roi_25d_shape_h": stack.shape[1],
                "roi_25d_shape_w": stack.shape[2],
                "roi_25d_window_min_hu": args.window_min,
                "roi_25d_window_max_hu": args.window_max,
                "roi_25d_preprocessing_version": PREPROCESSING_VERSION,
            }
            for key, value in values.items():
                manifest.loc[index, key] = value
            print(f"[{patient_id}] {status}: shape={stack.shape}")
        except Exception as error:
            manifest.loc[index, "roi_25d_status"] = "failed"
            manifest.loc[index, "roi_25d_message"] = str(error)
            print(f"[{patient_id}] failed: {error}")
        finally:
            write_manifest(manifest, args.manifest)


if __name__ == "__main__":
    main()

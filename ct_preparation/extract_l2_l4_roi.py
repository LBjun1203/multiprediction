#!/usr/bin/env python3
"""Crop a full-x/y L2-L4 CT slab, resample it, and locate L3."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from ct_preparation.common import (
    atomic_write_csv,
    clean_patient_id,
    orient_lps,
    patient_directory,
    relative_manifest_path,
    resample_ct,
    resolve_manifest_path,
)
from ct_preparation.manifest import SUCCESS_STATUSES, load_manifest, write_manifest

ROI_COLUMNS = (
    "l2_l4_roi_path",
    "l2_l4_roi_shape_x",
    "l2_l4_roi_shape_y",
    "l2_l4_roi_shape_z",
    "l2_l4_roi_spacing_x",
    "l2_l4_roi_spacing_y",
    "l2_l4_roi_spacing_z",
    "l3_center_index",
    "l3_centroid_x_mm",
    "l3_centroid_y_mm",
    "l3_centroid_z_mm",
    "l2_l4_roi_status",
    "l2_l4_roi_message",
)


def write_outputs(manifest, manifest_path: Path, metadata_path: Path) -> None:
    write_manifest(manifest, manifest_path)
    atomic_write_csv(manifest.loc[:, ["patient_id", *ROI_COLUMNS]], metadata_path)


def geometry_matches(reference: sitk.Image, other: sitk.Image) -> bool:
    return (
        reference.GetDimension() == other.GetDimension() == 3
        and reference.GetSize() == other.GetSize()
        and np.allclose(reference.GetSpacing(), other.GetSpacing(), rtol=0, atol=1e-5)
        and np.allclose(reference.GetOrigin(), other.GetOrigin(), rtol=0, atol=1e-4)
        and np.allclose(
            reference.GetDirection(), other.GetDirection(), rtol=0, atol=1e-5
        )
    )


def read_aligned_images(
    ct_path: Path, mask_paths: list[Path]
) -> tuple[sitk.Image, list[sitk.Image]]:
    if not ct_path.is_file():
        raise FileNotFoundError(f"CT does not exist: {ct_path}")
    ct = sitk.ReadImage(str(ct_path))
    if ct.GetDimension() != 3:
        raise ValueError(f"Expected a 3D CT, found {ct.GetDimension()}D")
    masks: list[sitk.Image] = []
    for mask_path in mask_paths:
        if not mask_path.is_file():
            raise FileNotFoundError(f"Mask does not exist: {mask_path}")
        mask = sitk.ReadImage(str(mask_path))
        if not geometry_matches(ct, mask):
            raise ValueError(f"Mask geometry is not aligned with CT: {mask_path.name}")
        if int(np.count_nonzero(sitk.GetArrayViewFromImage(mask))) <= 0:
            raise ValueError(f"Mask is empty: {mask_path.name}")
        masks.append(mask)
    return ct, masks


def build_roi(
    ct_path: Path,
    mask_paths: list[Path],
    output_path: Path,
    target_spacing: tuple[float, float, float],
    z_margin_mm: float,
) -> dict[str, float | int]:
    ct, masks = read_aligned_images(ct_path, mask_paths)

    # DICOM LPS makes SimpleITK index axis 2 the inferior-superior axis.  After
    # GetArrayFromImage, NumPy axis 0 is therefore the stable axial-slice axis.
    ct_lps = orient_lps(ct)
    masks_lps = [sitk.Cast(orient_lps(mask) > 0, sitk.sitkUInt8) for mask in masks]
    orientation = sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
        ct_lps.GetDirection()
    )
    if orientation != "LPS":
        raise RuntimeError(
            f"Orientation normalization failed: expected LPS, found {orientation}"
        )

    union_array = np.logical_or.reduce(
        [sitk.GetArrayViewFromImage(mask) > 0 for mask in masks_lps]
    )
    occupied_z = np.flatnonzero(union_array.any(axis=(1, 2)))
    if occupied_z.size == 0:
        raise ValueError("L2/L3/L4 union mask is empty")
    margin_slices = math.ceil(z_margin_mm / ct_lps.GetSpacing()[2])
    z_start = max(0, int(occupied_z[0]) - margin_slices)
    z_stop = min(ct_lps.GetSize()[2], int(occupied_z[-1]) + margin_slices + 1)

    crop_size = [ct_lps.GetSize()[0], ct_lps.GetSize()[1], z_stop - z_start]
    cropped = sitk.RegionOfInterest(ct_lps, crop_size, [0, 0, z_start])

    l3_stats = sitk.LabelShapeStatisticsImageFilter()
    l3_stats.Execute(masks_lps[1])
    if not l3_stats.HasLabel(1):
        raise ValueError("L3 mask has no foreground label")
    centroid = tuple(float(value) for value in l3_stats.GetCentroid(1))

    roi = resample_ct(cropped, target_spacing)
    continuous_index = roi.TransformPhysicalPointToContinuousIndex(centroid)
    center_slice = round(continuous_index[2])
    if not (0 <= center_slice < roi.GetSize()[2]):
        raise ValueError(
            f"L3 centroid maps outside resampled ROI: slice={center_slice}, depth={roi.GetSize()[2]}"
        )
    if not np.isfinite(sitk.GetArrayViewFromImage(roi)).all():
        raise ValueError("Resampled ROI contains non-finite voxel values")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name("l2_l4_roi.partial.nii.gz")
    sitk.WriteImage(roi, str(temporary), True)
    temporary.replace(output_path)

    size = roi.GetSize()
    spacing = roi.GetSpacing()
    return {
        "l2_l4_roi_shape_x": int(size[0]),
        "l2_l4_roi_shape_y": int(size[1]),
        "l2_l4_roi_shape_z": int(size[2]),
        "l2_l4_roi_spacing_x": float(spacing[0]),
        "l2_l4_roi_spacing_y": float(spacing[1]),
        "l2_l4_roi_spacing_z": float(spacing[2]),
        "l3_center_index": center_slice,
        "l3_centroid_x_mm": centroid[0],
        "l3_centroid_y_mm": centroid[1],
        "l3_centroid_z_mm": centroid[2],
    }


def validate_existing_roi(path: Path, center_slice: object) -> dict[str, float | int]:
    roi = sitk.ReadImage(str(path))
    if roi.GetDimension() != 3:
        raise ValueError(f"Expected a 3D ROI, found {roi.GetDimension()}D")
    orientation = sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
        roi.GetDirection()
    )
    if orientation != "LPS":
        raise ValueError(f"Existing ROI is not LPS-oriented: {orientation}")
    center = int(float(center_slice))
    if not (0 <= center < roi.GetSize()[2]):
        raise ValueError(f"Existing L3 center slice is outside ROI: {center}")
    if not np.isfinite(sitk.GetArrayViewFromImage(roi)).all():
        raise ValueError("Existing ROI contains non-finite voxel values")
    size = roi.GetSize()
    spacing = roi.GetSpacing()
    return {
        "l2_l4_roi_shape_x": int(size[0]),
        "l2_l4_roi_shape_y": int(size[1]),
        "l2_l4_roi_shape_z": int(size[2]),
        "l2_l4_roi_spacing_x": float(spacing[0]),
        "l2_l4_roi_spacing_y": float(spacing[1]),
        "l2_l4_roi_spacing_z": float(spacing[2]),
        "l3_center_index": center,
    }


def select_indices(manifest, patient_id: str | None, limit: int | None) -> list[int]:
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
    parser.add_argument("--roi-metadata", type=Path)
    parser.add_argument(
        "--target-spacing", nargs=3, type=float, default=(1.5, 1.5, 2.0)
    )
    parser.add_argument("--z-margin-mm", type=float, default=20.0)
    parser.add_argument("--patient-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Store generated paths relative to --output-root for portability",
    )
    args = parser.parse_args()

    target_spacing = tuple(float(value) for value in args.target_spacing)
    if any(value <= 0 or not math.isfinite(value) for value in target_spacing):
        raise ValueError("--target-spacing values must be finite and positive")
    if args.z_margin_mm < 0 or not math.isfinite(args.z_margin_mm):
        raise ValueError("--z-margin-mm must be finite and non-negative")
    args.output_root.mkdir(parents=True, exist_ok=True)
    metadata_path = args.roi_metadata or args.output_root / "roi_metadata.csv"
    manifest = load_manifest(args.manifest)

    for index in select_indices(manifest, args.patient_id, args.limit):
        patient_id = str(manifest.loc[index, "patient_id"])
        print(f"[{patient_id}] L2-L4 ROI")
        try:
            if (
                str(manifest.loc[index, "segmentation_status"]).lower()
                not in SUCCESS_STATUSES
            ):
                manifest.loc[index, "l2_l4_roi_status"] = "skipped"
                manifest.loc[index, "l2_l4_roi_message"] = (
                    "segmentation did not succeed"
                )
                print(f"[{patient_id}] skipped: segmentation did not succeed")
                continue
            output_path = (
                patient_directory(args.output_root, patient_id) / "l2_l4_roi.nii.gz"
            )
            if args.skip_existing and output_path.is_file():
                metadata = validate_existing_roi(
                    output_path, manifest.loc[index, "l3_center_index"]
                )
                status = "skipped_existing"
                message = "existing ROI and L3 center slice passed basic checks"
            else:
                metadata = build_roi(
                    resolve_manifest_path(
                        manifest.loc[index, "full_ct_path"], args.output_root
                    ),
                    [
                        resolve_manifest_path(
                            manifest.loc[index, f"{level.lower()}_mask_path"],
                            args.output_root,
                        )
                        for level in ("L2", "L3", "L4")
                    ],
                    output_path,
                    target_spacing,
                    args.z_margin_mm,
                )
                status = "success"
                message = (
                    f"LPS orientation; full x/y FOV; L2-L4 union + {args.z_margin_mm:g} mm z margin; "
                    f"linear resampling to {target_spacing} mm"
                )
            manifest.loc[index, "l2_l4_roi_path"] = relative_manifest_path(
                output_path, args.output_root, args.relative_paths
            )
            for key, value in metadata.items():
                manifest.loc[index, key] = value
            manifest.loc[index, "l2_l4_roi_status"] = status
            manifest.loc[index, "l2_l4_roi_message"] = message
            print(
                f"[{patient_id}] {status}: shape=({metadata['l2_l4_roi_shape_x']}, "
                f"{metadata['l2_l4_roi_shape_y']}, {metadata['l2_l4_roi_shape_z']}), "
                f"spacing=({metadata['l2_l4_roi_spacing_x']}, {metadata['l2_l4_roi_spacing_y']}, "
                f"{metadata['l2_l4_roi_spacing_z']}), L3 slice={metadata['l3_center_index']}"
            )
        except Exception as error:  # noqa: BLE001 - isolate one bad patient
            manifest.loc[index, "l2_l4_roi_status"] = "failed"
            manifest.loc[index, "l2_l4_roi_message"] = str(error)
            print(f"[{patient_id}] failed: {error}")
        finally:
            write_outputs(manifest, args.manifest, metadata_path)

    print(f"Manifest: {args.manifest.resolve()}")
    print(f"ROI metadata: {metadata_path.resolve()}")


if __name__ == "__main__":
    main()

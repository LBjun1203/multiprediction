#!/usr/bin/env python3
"""Build fixed-shape, L3-centered L2-L4 ROI 3D tensors.

The builder consumes ``l2_l4_roi`` images only. It normalizes physical spacing,
samples a fixed physical field of view around the L3 centroid, pads outside the
source ROI with air (-1024 HU), windows HU values, and stores ``[D,H,W]``
``float32`` arrays in ``[0,1]``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from .common import (
    AIR_HU,
    clean_patient_id,
    fixed_physical_crop_pad,
    orient_lps,
    patient_directory,
    relative_manifest_path,
    resample_ct,
    resolve_manifest_path,
    validate_model_array,
    validate_spacing,
    window_hu,
)
from .manifest import load_manifest, write_manifest

PREPROCESSING_VERSION = "roi3d-l2l4-v1"
DEFAULT_SPACING_XYZ = (1.5, 1.5, 2.0)
DEFAULT_SHAPE_DHW = (64, 160, 160)
DEFAULT_WINDOW_HU = (-200.0, 300.0)


def _l3_center_lps(image: sitk.Image, row: object) -> tuple[float, float, float]:
    centroid_values = [getattr(row, f"l3_centroid_{axis}_mm", "") for axis in "xyz"]
    try:
        centroid = tuple(float(value) for value in centroid_values)
    except (TypeError, ValueError):
        centroid = ()
    if len(centroid) == 3 and np.isfinite(centroid).all():
        return centroid
    center_index = int(float(row.l3_center_index))
    size = image.GetSize()
    if not 0 <= center_index < size[2]:
        raise ValueError(
            f"l3_center_index={center_index} is outside ROI depth={size[2]}"
        )
    return image.TransformContinuousIndexToPhysicalPoint(
        ((size[0] - 1) / 2.0, (size[1] - 1) / 2.0, float(center_index))
    )


def build_roi_3d_array(
    roi: sitk.Image,
    center_lps_mm: tuple[float, float, float],
    target_spacing_xyz: tuple[float, float, float] = DEFAULT_SPACING_XYZ,
    shape_dhw: tuple[int, int, int] = DEFAULT_SHAPE_DHW,
    window_hu_bounds: tuple[float, float] = DEFAULT_WINDOW_HU,
) -> tuple[np.ndarray, sitk.Image]:
    if roi.GetDimension() != 3:
        raise ValueError(f"Expected a 3D L2-L4 ROI, found {roi.GetDimension()}D")
    spacing = validate_spacing(target_spacing_xyz)
    oriented = orient_lps(roi)
    normalized = resample_ct(oriented, spacing, default_value=AIR_HU)
    cropped = fixed_physical_crop_pad(normalized, center_lps_mm, shape_dhw, AIR_HU)
    hu = sitk.GetArrayFromImage(cropped)
    array = window_hu(hu, *window_hu_bounds)
    validate_model_array(array, shape_dhw)
    return array, cropped


def save_preview(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    d, h, w = array.shape
    planes = (
        (array[d // 2], "axial"),
        (array[:, h // 2, :], "coronal"),
        (array[:, :, w // 2], "sagittal"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, (plane, title) in zip(axes, planes):
        axis.imshow(plane, cmap="gray", vmin=0.0, vmax=1.0, origin="lower")
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_array(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as file:
        np.save(file, array, allow_pickle=False)
    temporary.replace(path)


def _indices(manifest, patient_id: str | None, limit: int | None) -> list[int]:
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
    parser.add_argument(
        "--target-spacing", nargs=3, type=float, default=DEFAULT_SPACING_XYZ
    )
    parser.add_argument(
        "--shape", nargs=3, type=int, default=DEFAULT_SHAPE_DHW, metavar=("D", "H", "W")
    )
    parser.add_argument(
        "--window",
        nargs=2,
        type=float,
        default=DEFAULT_WINDOW_HU,
        metavar=("MIN_HU", "MAX_HU"),
    )
    parser.add_argument("--patient-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--relative-paths", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    spacing = validate_spacing(args.target_spacing)
    shape = tuple(int(value) for value in args.shape)
    window = tuple(float(value) for value in args.window)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)

    for index in _indices(manifest, args.patient_id, args.limit):
        patient_id = str(manifest.loc[index, "patient_id"])
        try:
            if str(manifest.loc[index, "l2_l4_roi_status"]).lower() not in {
                "success",
                "skipped_existing",
            }:
                raise ValueError("l2_l4_roi did not succeed")
            roi_path = resolve_manifest_path(
                manifest.loc[index, "l2_l4_roi_path"], args.output_root
            )
            output = patient_directory(args.output_root, patient_id) / "roi_3d.npy"
            preview = (
                patient_directory(args.output_root, patient_id) / "roi_3d_preview.png"
            )
            row = next(manifest.loc[[index]].itertuples(index=False))
            roi = sitk.ReadImage(str(roi_path))
            center = _l3_center_lps(roi, row)
            if args.skip_existing and output.is_file():
                array = np.load(output, allow_pickle=False)
                validate_model_array(array, shape)
                status = "skipped_existing"
            else:
                array, _ = build_roi_3d_array(roi, center, spacing, shape, window)
                _write_array(array, output)
                if not args.no_preview:
                    save_preview(array, preview)
                status = "success"
            d, h, w = shape
            metadata = {
                "roi_3d_path": relative_manifest_path(
                    output, args.output_root, args.relative_paths
                ),
                "roi_3d_status": status,
                "roi_3d_message": "LPS; spacing-normalized fixed physical crop centered at L3; air padding",
                "roi_3d_shape_d": d,
                "roi_3d_shape_h": h,
                "roi_3d_shape_w": w,
                "roi_3d_spacing_x": spacing[0],
                "roi_3d_spacing_y": spacing[1],
                "roi_3d_spacing_z": spacing[2],
                "roi_3d_crop_extent_x_mm": w * spacing[0],
                "roi_3d_crop_extent_y_mm": h * spacing[1],
                "roi_3d_crop_extent_z_mm": d * spacing[2],
                "roi_3d_window_min_hu": window[0],
                "roi_3d_window_max_hu": window[1],
                "roi_3d_dtype": str(array.dtype),
                "roi_3d_value_min": float(array.min()),
                "roi_3d_value_max": float(array.max()),
                "roi_3d_preprocessing_version": PREPROCESSING_VERSION,
            }
            for key, value in metadata.items():
                manifest.loc[index, key] = value
            print(f"[{patient_id}] {status}: shape={shape}, spacing={spacing}")
        except Exception as error:  # isolate one bad patient
            manifest.loc[index, "roi_3d_status"] = "failed"
            manifest.loc[index, "roi_3d_message"] = str(error)
            print(f"[{patient_id}] failed: {error}")
        finally:
            write_manifest(manifest, args.manifest)


if __name__ == "__main__":
    main()

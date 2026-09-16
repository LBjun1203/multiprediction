#!/usr/bin/env python3
"""Build a non-trainable full-volume 3D array boundary from ``full_ct``.

Depth remains variable. Training rejects this representation until a padding,
masking, and pooling policy is approved.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from ct_preparation.common import (
    orient_lps,
    patient_directory,
    relative_manifest_path,
    resample_ct,
    resolve_manifest_path,
    window_hu,
)
from ct_preparation.manifest import SUCCESS_STATUSES, load_manifest, write_manifest

PREPROCESSING_VERSION = "full3d-v1"


def build_full_3d_array(
    image: sitk.Image,
    spacing_xyz: tuple[float, float, float],
    window: tuple[float, float],
) -> np.ndarray:
    normalized = resample_ct(orient_lps(image), spacing_xyz)
    array = window_hu(sitk.GetArrayFromImage(normalized), *window)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError("full_3d must be a finite [D,H,W] array")
    return array


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--target-spacing", nargs=3, type=float, default=(1.5, 1.5, 2.0)
    )
    parser.add_argument("--window", nargs=2, type=float, default=(-200.0, 300.0))
    parser.add_argument("--patient-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--relative-paths", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    manifest = load_manifest(args.manifest)
    selected = manifest
    if args.patient_id is not None:
        selected = selected.loc[selected["patient_id"] == str(args.patient_id)]
        if selected.empty:
            raise ValueError(f"patient_id={args.patient_id!r} is not present")
    if args.limit is not None:
        selected = selected.head(args.limit)
    for index in selected.index:
        patient_id = str(manifest.loc[index, "patient_id"])
        output = patient_directory(args.output_root, patient_id) / "full_3d.npy"
        try:
            if (
                str(manifest.loc[index, "full_ct_status"]).lower()
                not in SUCCESS_STATUSES
            ):
                raise ValueError("full_ct did not succeed")
            if args.skip_existing and output.is_file():
                array = np.load(output, allow_pickle=False)
                status = "skipped_existing"
            else:
                source = resolve_manifest_path(
                    manifest.loc[index, "full_ct_path"], args.output_root
                )
                array = build_full_3d_array(
                    sitk.ReadImage(str(source)),
                    tuple(args.target_spacing),
                    tuple(args.window),
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_suffix(".npy.partial")
                with temporary.open("wb") as file:
                    np.save(file, array, allow_pickle=False)
                temporary.replace(output)
                status = "success"
            manifest.loc[index, "full_3d_path"] = relative_manifest_path(
                output, args.output_root, args.relative_paths
            )
            manifest.loc[index, "full_3d_status"] = status
            manifest.loc[index, "full_3d_message"] = (
                "variable-depth full coverage; not trainable"
            )
            manifest.loc[index, "full_3d_shape_d"] = array.shape[0]
            manifest.loc[index, "full_3d_shape_h"] = array.shape[1]
            manifest.loc[index, "full_3d_shape_w"] = array.shape[2]
            manifest.loc[index, "full_3d_preprocessing_version"] = PREPROCESSING_VERSION
            print(f"[{patient_id}] {status}: shape={array.shape}")
        except Exception as error:
            manifest.loc[index, "full_3d_status"] = "failed"
            manifest.loc[index, "full_3d_message"] = str(error)
            print(f"[{patient_id}] failed: {error}")
        finally:
            write_manifest(manifest, args.manifest)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run TotalSegmentator for L2/L3/L4 and record numerical QC."""

from __future__ import annotations

import argparse
import shutil
import subprocess
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

LABELS = ("vertebrae_L2", "vertebrae_L3", "vertebrae_L4")
SEGMENTATION_COLUMNS = (
    "l2_mask_path",
    "l3_mask_path",
    "l4_mask_path",
    "l2_voxels",
    "l3_voxels",
    "l4_voxels",
    "segmentation_status",
    "segmentation_message",
)


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


def select_indices(manifest, patient_id: str | None, limit: int | None) -> list[int]:
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
            if (
                str(manifest.loc[index, "full_ct_status"]).lower()
                not in SUCCESS_STATUSES
            ):
                manifest.loc[index, "segmentation_status"] = "skipped"
                manifest.loc[index, "segmentation_message"] = "full_ct did not succeed"
                print(f"[{patient_id}] skipped: full_ct did not succeed")
                continue
            ct_path = resolve_manifest_path(
                manifest.loc[index, "full_ct_path"], args.output_root
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
                manifest.loc[index, f"{short.lower()}_mask_path"] = (
                    relative_manifest_path(
                        segmentation_dir / f"{label}.nii.gz",
                        args.output_root,
                        args.relative_paths,
                    )
                )
                manifest.loc[index, f"{short.lower()}_voxels"] = counts[label]
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

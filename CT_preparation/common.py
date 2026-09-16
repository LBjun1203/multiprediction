"""Small shared primitives for CT preparation stages.

SimpleITK image geometry is always expressed as ``(x, y, z)``. NumPy model
tensors use ``(d, h, w)`` (or ``(c, h, w)`` for 2.5D inputs).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

AIR_HU = -1024.0


def clean_patient_id(value: object) -> str:
    if pd.isna(value):
        raise ValueError("patient_id contains a missing value")
    text = str(value).strip()
    if not text:
        raise ValueError("patient_id contains an empty value")
    match = re.fullmatch(r"(\d+)\.0", text)
    return match.group(1) if match else text


def patient_directory(output_root: Path, patient_id: str) -> Path:
    cleaned = clean_patient_id(patient_id)
    if cleaned in {".", ".."} or Path(cleaned).name != cleaned:
        raise ValueError(f"Unsafe patient_id for an output directory: {patient_id!r}")
    return output_root / cleaned


def resolve_manifest_path(value: object, base_directory: Path) -> Path:
    path = Path(str(value).strip())
    return path if path.is_absolute() else base_directory / path


def relative_manifest_path(path: Path, output_root: Path, relative: bool) -> str:
    resolved = path.resolve()
    if not relative:
        return str(resolved)
    try:
        return str(resolved.relative_to(output_root.resolve()))
    except ValueError as error:
        raise ValueError(f"Output path is outside --output-root: {resolved}") from error


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def orient_lps(image: sitk.Image) -> sitk.Image:
    oriented = sitk.DICOMOrient(image, "LPS")
    code = sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
        oriented.GetDirection()
    )
    if code != "LPS":
        raise RuntimeError(f"LPS normalization failed: found {code}")
    return oriented


def validate_spacing(spacing_xyz: Iterable[float]) -> tuple[float, float, float]:
    spacing = tuple(float(value) for value in spacing_xyz)
    if len(spacing) != 3 or any(
        value <= 0 or not math.isfinite(value) for value in spacing
    ):
        raise ValueError(
            f"spacing must contain three finite positive values, got {spacing}"
        )
    return spacing


def resample_ct(
    image: sitk.Image,
    target_spacing_xyz: Iterable[float],
    default_value: float = AIR_HU,
) -> sitk.Image:
    spacing = validate_spacing(target_spacing_xyz)
    old_size = image.GetSize()
    old_spacing = image.GetSpacing()
    new_size = [
        max(1, round((old_size[axis] - 1) * old_spacing[axis] / spacing[axis]) + 1)
        for axis in range(3)
    ]
    return sitk.Resample(
        sitk.Cast(image, sitk.sitkFloat32),
        new_size,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkLinear,
        image.GetOrigin(),
        spacing,
        image.GetDirection(),
        float(default_value),
        sitk.sitkFloat32,
    )


def fixed_physical_crop_pad(
    image: sitk.Image,
    center_lps_mm: Iterable[float],
    shape_dhw: Iterable[int],
    pad_hu: float = AIR_HU,
) -> sitk.Image:
    """Sample a fixed physical field of view centered at an LPS point.

    The input must already have the desired spacing. Values outside its physical
    extent are filled with ``pad_hu``; anatomical scale is never resized.
    """
    shape = tuple(int(value) for value in shape_dhw)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError(f"shape_dhw must contain three positive integers, got {shape}")
    center = tuple(float(value) for value in center_lps_mm)
    if len(center) != 3 or not np.isfinite(center).all():
        raise ValueError(f"center_lps_mm must be a finite (x,y,z) point, got {center}")
    size_xyz = (shape[2], shape[1], shape[0])
    center_index = image.TransformPhysicalPointToContinuousIndex(center)
    start_index = tuple(
        center_index[axis] - (size_xyz[axis] - 1) / 2.0 for axis in range(3)
    )
    origin = image.TransformContinuousIndexToPhysicalPoint(start_index)
    return sitk.Resample(
        sitk.Cast(image, sitk.sitkFloat32),
        size_xyz,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkLinear,
        origin,
        image.GetSpacing(),
        image.GetDirection(),
        float(pad_hu),
        sitk.sitkFloat32,
    )


def window_hu(array: np.ndarray, lower: float, upper: float) -> np.ndarray:
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError(f"invalid HU window: ({lower}, {upper})")
    result = np.clip(array.astype(np.float32, copy=False), lower, upper)
    result = (result - lower) / (upper - lower)
    return result.astype(np.float32, copy=False)


def validate_model_array(array: np.ndarray, expected_shape: Iterable[int]) -> None:
    shape = tuple(int(value) for value in expected_shape)
    if array.shape != shape:
        raise ValueError(f"array shape is {array.shape}, expected {shape}")
    if array.dtype != np.float32:
        raise ValueError(f"array dtype must be float32, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError("array contains non-finite values")
    minimum = float(array.min())
    maximum = float(array.max())
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(f"array range must be [0, 1], got [{minimum}, {maximum}]")

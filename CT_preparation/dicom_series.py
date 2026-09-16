"""Discover and select one diagnostic axial CT DICOM series."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

LOCALIZER_PATTERN = re.compile(
    r"locali[sz]er|scout|topogram|surview|scanogram|定位", re.IGNORECASE
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
        return (self.coverage_mm, len(self.files), self.uid)


def _metadata_value(reader: sitk.ImageFileReader, key: str) -> str:
    return reader.GetMetaData(key).strip() if reader.HasMetaDataKey(key) else ""


def read_dicom_metadata(path: str) -> dict[str, str]:
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    return {
        "modality": _metadata_value(reader, "0008|0060"),
        "description": _metadata_value(reader, "0008|103e"),
        "image_type": _metadata_value(reader, "0008|0008"),
        "position": _metadata_value(reader, "0020|0032"),
        "orientation": _metadata_value(reader, "0020|0037"),
    }


def _parse_vector(text: str, length: int) -> np.ndarray | None:
    try:
        values = np.asarray([float(item) for item in text.split("\\")], dtype=float)
    except ValueError:
        return None
    return values if values.shape == (length,) and np.isfinite(values).all() else None


def _series_geometry(first: dict[str, str], last: dict[str, str]) -> tuple[bool, float]:
    orientation = _parse_vector(first["orientation"], 6)
    first_position = _parse_vector(first["position"], 3)
    last_position = _parse_vector(last["position"], 3)
    if orientation is None:
        return False, 0.0
    normal = np.cross(orientation[:3], orientation[3:])
    norm = float(np.linalg.norm(normal))
    if norm <= 0 or not math.isfinite(norm):
        return False, 0.0
    normal /= norm
    axial = abs(float(normal[2])) >= 0.8
    coverage = (
        abs(float(np.dot(last_position - first_position, normal)))
        if first_position is not None and last_position is not None
        else 0.0
    )
    return axial, coverage


def discover_series(
    patient_directory: Path, min_slices: int = 16
) -> list[SeriesCandidate]:
    if min_slices <= 0:
        raise ValueError("min_slices must be positive")
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(patient_directory)) or []
    candidates: list[SeriesCandidate] = []
    for uid in series_ids:
        files = tuple(
            sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(patient_directory), uid)
        )
        if len(files) < min_slices:
            continue
        first = read_dicom_metadata(files[0])
        if first["modality"].upper() != "CT":
            continue
        if LOCALIZER_PATTERN.search(f"{first['description']} {first['image_type']}"):
            continue
        axial, coverage = _series_geometry(first, read_dicom_metadata(files[-1]))
        if axial:
            candidates.append(
                SeriesCandidate(
                    uid=str(uid),
                    files=files,
                    description=first["description"],
                    image_type=first["image_type"],
                    coverage_mm=coverage,
                )
            )
    return candidates


def select_series(
    patient_directory: Path, min_slices: int = 16
) -> tuple[SeriesCandidate, list[SeriesCandidate]]:
    candidates = discover_series(patient_directory, min_slices)
    if not candidates:
        raise ValueError(
            "No diagnostic axial CT series remained after modality, localizer, "
            f"orientation, and minimum-slice ({min_slices}) checks"
        )
    return max(candidates, key=lambda item: item.score), candidates


def read_series(candidate: SeriesCandidate) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(candidate.files)
    image = reader.Execute()
    if image.GetDimension() != 3:
        raise ValueError(f"Expected a 3D CT, found {image.GetDimension()}D")
    if any(value <= 0 or not math.isfinite(value) for value in image.GetSpacing()):
        raise ValueError(f"Invalid CT spacing: {image.GetSpacing()}")
    return sitk.Cast(image, sitk.sitkFloat32)

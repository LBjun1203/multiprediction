"""Canonical wide CT manifest schema and representation registry."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .common import atomic_write_csv, clean_patient_id, resolve_manifest_path

MANIFEST_SCHEMA_VERSION = "1"
SUCCESS_STATUSES = frozenset({"success", "skipped_existing"})
LEGACY_ALIASES = {
    "nifti_path": "full_ct_path",
    "conversion_status": "full_ct_status",
    "roi_path": "l2_l4_roi_path",
    "roi_status": "l2_l4_roi_status",
    "L3_center_slice": "l3_center_index",
    "npy_25d_path": "roi_25d_path",
    "build_25d_status": "roi_25d_status",
    "L2_mask_path": "l2_mask_path",
    "L3_mask_path": "l3_mask_path",
    "L4_mask_path": "l4_mask_path",
    "L3_centroid_x_mm": "l3_centroid_x_mm",
    "L3_centroid_y_mm": "l3_centroid_y_mm",
    "L3_centroid_z_mm": "l3_centroid_z_mm",
}

CANONICAL_COLUMNS = (
    "manifest_schema_version",
    "patient_id",
    "source_ct_path",
    "full_ct_path",
    "full_ct_status",
    "full_ct_message",
    "selected_series_uid",
    "series_description",
    "full_ct_shape_x",
    "full_ct_shape_y",
    "full_ct_shape_z",
    "full_ct_spacing_x",
    "full_ct_spacing_y",
    "full_ct_spacing_z",
    "l2_mask_path",
    "l3_mask_path",
    "l4_mask_path",
    "segmentation_status",
    "segmentation_message",
    "l2_l4_roi_path",
    "l2_l4_roi_status",
    "l2_l4_roi_message",
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
    "roi_25d_path",
    "roi_25d_status",
    "roi_25d_message",
    "roi_25d_shape_c",
    "roi_25d_shape_h",
    "roi_25d_shape_w",
    "roi_25d_window_min_hu",
    "roi_25d_window_max_hu",
    "roi_25d_preprocessing_version",
    "roi_3d_path",
    "roi_3d_status",
    "roi_3d_message",
    "roi_3d_shape_d",
    "roi_3d_shape_h",
    "roi_3d_shape_w",
    "roi_3d_spacing_x",
    "roi_3d_spacing_y",
    "roi_3d_spacing_z",
    "roi_3d_crop_extent_x_mm",
    "roi_3d_crop_extent_y_mm",
    "roi_3d_crop_extent_z_mm",
    "roi_3d_window_min_hu",
    "roi_3d_window_max_hu",
    "roi_3d_dtype",
    "roi_3d_value_min",
    "roi_3d_value_max",
    "roi_3d_preprocessing_version",
    "full_3d_path",
    "full_3d_status",
    "full_3d_message",
    "full_3d_shape_d",
    "full_3d_shape_h",
    "full_3d_shape_w",
    "full_3d_preprocessing_version",
)


@dataclass(frozen=True)
class RepresentationRecord:
    path_column: str
    status_column: str
    preprocessing_column: str
    stored_layout: str
    sample_layout: str
    compatible_encoders: tuple[str, ...]
    trainable: bool


REPRESENTATIONS = {
    "roi_25d": RepresentationRecord(
        "roi_25d_path",
        "roi_25d_status",
        "roi_25d_preprocessing_version",
        "CHW",
        "CHW",
        ("resnet18_2d",),
        True,
    ),
    "roi_3d": RepresentationRecord(
        "roi_3d_path",
        "roi_3d_status",
        "roi_3d_preprocessing_version",
        "DHW",
        "CDHW",
        ("conv3d",),
        True,
    ),
    "full_3d": RepresentationRecord(
        "full_3d_path",
        "full_3d_status",
        "full_3d_preprocessing_version",
        "DHW",
        "CDHW",
        ("none",),
        False,
    ),
}


@dataclass(frozen=True)
class CTInputSpec:
    representation: str
    encoder: str
    expected_shape: tuple[int, ...]
    preprocessing_version: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["expected_shape"] = list(self.expected_shape)
        return value


LEGACY_CT_INPUT_SPEC = CTInputSpec(
    representation="roi_25d",
    encoder="resnet18_2d",
    expected_shape=(3, 224, 224),
    preprocessing_version="roi25d-v1",
)


def resolve_ct_input_spec(
    value: CTInputSpec | Mapping[str, Any] | None,
    *,
    require_trainable: bool = True,
) -> CTInputSpec:
    if value is None:
        spec = LEGACY_CT_INPUT_SPEC
    elif isinstance(value, CTInputSpec):
        spec = value
    else:
        required = {
            "representation",
            "encoder",
            "expected_shape",
            "preprocessing_version",
        }
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"CT input configuration is missing keys: {missing}")
        raw_shape = value["expected_shape"]
        if not isinstance(raw_shape, (list, tuple)):
            raise ValueError("expected_shape must be a list or tuple")
        if any(
            not isinstance(item, int) or isinstance(item, bool) for item in raw_shape
        ):
            raise ValueError("expected_shape must contain integers")
        spec = CTInputSpec(
            representation=str(value["representation"]),
            encoder=str(value["encoder"]),
            expected_shape=tuple(int(item) for item in raw_shape),
            preprocessing_version=str(value["preprocessing_version"]),
        )
    record = REPRESENTATIONS.get(spec.representation)
    if record is None:
        raise ValueError(
            f"Unknown CT representation {spec.representation!r}; "
            f"choose one of {sorted(REPRESENTATIONS)}"
        )
    if require_trainable and not record.trainable:
        raise ValueError(
            f"CT representation {spec.representation!r} is an organized preprocessing "
            "boundary but is not registered as a trainable input"
        )
    if spec.encoder not in record.compatible_encoders:
        raise ValueError(
            f"Encoder {spec.encoder!r} is incompatible with {spec.representation!r}; "
            f"allowed encoders: {list(record.compatible_encoders)}"
        )
    if not spec.preprocessing_version.strip():
        raise ValueError("preprocessing_version must be non-empty")
    if any(
        not isinstance(item, int) or isinstance(item, bool) or item <= 0
        for item in spec.expected_shape
    ):
        raise ValueError(
            f"expected_shape must contain positive integers: {spec.expected_shape}"
        )
    if spec.representation == "roi_25d":
        if len(spec.expected_shape) != 3 or spec.expected_shape[0] != 3:
            raise ValueError("roi_25d expected_shape must be [3,H,W]")
        if min(spec.expected_shape[1:]) < 32:
            raise ValueError("roi_25d height and width must be at least 32")
    elif spec.representation == "roi_3d":
        if len(spec.expected_shape) != 4 or spec.expected_shape[0] != 1:
            raise ValueError("roi_3d expected_shape must be [1,D,H,W]")
        if min(spec.expected_shape[1:]) < 8:
            raise ValueError("roi_3d depth, height, and width must be at least 8")
    return spec


def canonicalize_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    manifest = frame.copy()
    for legacy, canonical in LEGACY_ALIASES.items():
        if canonical not in manifest.columns and legacy in manifest.columns:
            manifest[canonical] = manifest[legacy]
    missing = sorted({"patient_id"} - set(manifest.columns))
    if missing:
        raise ValueError(f"CT manifest is missing required columns: {missing}")
    manifest["patient_id"] = manifest["patient_id"].map(clean_patient_id)
    if manifest["patient_id"].duplicated().any():
        duplicate = manifest.loc[
            manifest["patient_id"].duplicated(), "patient_id"
        ].iloc[0]
        raise ValueError(f"CT manifest has duplicate patient_id: {duplicate!r}")
    for column in CANONICAL_COLUMNS:
        if column not in manifest.columns:
            manifest[column] = ""
    manifest["manifest_schema_version"] = manifest["manifest_schema_version"].replace(
        "", MANIFEST_SCHEMA_VERSION
    )
    return manifest


def load_manifest(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"CT manifest does not exist: {source}")
    frame = pd.read_csv(source, dtype={"patient_id": "string"}, keep_default_na=False)
    return canonicalize_manifest(frame)


def write_manifest(frame: pd.DataFrame, path: str | Path) -> None:
    manifest = canonicalize_manifest(frame)
    original_columns = list(frame.columns)
    ordered = list(dict.fromkeys([*CANONICAL_COLUMNS, *original_columns]))
    atomic_write_csv(manifest.loc[:, ordered], Path(path))


def resolve_representation_paths(
    manifest: pd.DataFrame,
    spec: CTInputSpec,
    base_directory: str | Path,
    required_patient_ids: set[str] | None = None,
    check_files: bool = True,
) -> pd.DataFrame:
    spec = resolve_ct_input_spec(spec, require_trainable=False)
    record = REPRESENTATIONS[spec.representation]
    frame = canonicalize_manifest(manifest)
    status = frame[record.status_column].astype(str).str.strip().str.lower()
    paths = frame[record.path_column].astype(str).str.strip()
    usable = status.isin(SUCCESS_STATUSES)
    validate = usable
    if required_patient_ids is not None:
        validate = validate & frame["patient_id"].isin(required_patient_ids)
    empty = paths.str.lower().isin({"", "nan", "none", "<na>"})
    if (validate & empty).any():
        patient_id = frame.loc[validate & empty, "patient_id"].iloc[0]
        raise ValueError(
            f"Manifest marks patient_id={patient_id!r} successful for "
            f"{spec.representation}, but {record.path_column} is empty"
        )
    versions = frame[record.preprocessing_column].astype(str).str.strip()
    mismatch = validate & versions.ne("") & versions.ne(spec.preprocessing_version)
    if mismatch.any():
        patient_id = frame.loc[mismatch, "patient_id"].iloc[0]
        actual = frame.loc[mismatch, record.preprocessing_column].iloc[0]
        raise ValueError(
            f"{spec.representation} preprocessing version mismatch for "
            f"patient_id={patient_id!r}: manifest={actual!r}, config={spec.preprocessing_version!r}"
        )
    root = Path(base_directory)
    for index in frame.index[usable & ~empty]:
        resolved = resolve_manifest_path(paths.loc[index], root).resolve()
        frame.loc[index, record.path_column] = str(resolved)
    if check_files:
        for row in frame.loc[validate, ["patient_id", record.path_column]].itertuples(
            index=False
        ):
            if not Path(str(row[1])).is_file():
                raise FileNotFoundError(
                    f"Missing {spec.representation} array for patient_id={row[0]!r}: {row[1]}"
                )
    return frame


def representation_signature(manifest: pd.DataFrame, spec: CTInputSpec) -> str:
    spec = resolve_ct_input_spec(spec, require_trainable=False)
    record = REPRESENTATIONS[spec.representation]
    columns = [
        "patient_id",
        record.path_column,
        record.status_column,
        record.preprocessing_column,
    ]
    rows = manifest.loc[:, columns].astype(str).sort_values("patient_id", kind="stable")
    payload = {
        "representation": spec.representation,
        "preprocessing_version": spec.preprocessing_version,
        "records": rows.to_dict(orient="records"),
    }
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def assert_same_ordered_cohort(left: Iterable[str], right: Iterable[str]) -> None:
    left_ids = [clean_patient_id(value) for value in left]
    right_ids = [clean_patient_id(value) for value in right]
    if left_ids != right_ids:
        raise ValueError(
            "Ordered comparison cohorts differ; use the same patient intersection"
        )

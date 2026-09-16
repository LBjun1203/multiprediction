#!/usr/bin/env python3
"""Run read-only data, CT, split, and normalization checks before training."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ct_modal.config import load_ct_config
from ct_preparation.manifest import (
    LEGACY_CT_INPUT_SPEC,
    REPRESENTATIONS,
    resolve_ct_input_spec,
)
from multidataset.huashan_dataset import (
    CT_SUCCESS_STATUSES,
    build_normalization_metadata,
    load_ct_manifest,
    load_ct_tensor,
    load_longitudinal_table,
    load_normalization_stats,
    load_patient_split,
    merge_longitudinal_with_ct,
    validate_normalization_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--ct-manifest", required=True, type=Path)
    parser.add_argument("--ct-config", type=Path, help="CT representation YAML")
    parser.add_argument(
        "--split-file", type=Path, default=Path("multidataset/patient_split.csv")
    )
    parser.add_argument(
        "--stats-file", type=Path, default=Path("multidataset/normalization.json")
    )
    parser.add_argument("--sheet", default="model_long_0_2y")
    parser.add_argument("--ct-root", type=Path)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--skip-array-content-check",
        action="store_true",
        help="Only check array existence, not shape/dtype/range content",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image_size < 32:
        raise ValueError("--image-size must be at least 32")
    ct_spec = (
        load_ct_config(args.ct_config)
        if args.ct_config is not None
        else resolve_ct_input_spec(
            {
                **LEGACY_CT_INPUT_SPEC.to_dict(),
                "expected_shape": [3, args.image_size, args.image_size],
            }
        )
    )
    table = load_longitudinal_table(args.input, args.sheet)
    split = load_patient_split(args.split_file)
    table_ids = set(table["patient_id"])
    split_ids = set(split["patient_id"])
    if table_ids != split_ids:
        raise ValueError(
            f"Dataset/split patient mismatch: missing={len(table_ids - split_ids)}, "
            f"unknown={len(split_ids - table_ids)}"
        )
    split_map = split.set_index("patient_id")["split"]
    table = table.copy()
    table["split"] = table["patient_id"].map(split_map)
    manifest = load_ct_manifest(
        args.ct_manifest,
        ct_root=args.ct_root,
        required_patient_ids=table_ids,
        ct_config=ct_spec,
    )
    merged, join_summary = merge_longitudinal_with_ct(table, manifest, ct_spec)
    frames = {
        name: merged.loc[merged["split"] == name].reset_index(drop=True)
        for name in ("train", "val", "test")
    }
    if any(frame.empty for frame in frames.values()):
        empty = [name for name, frame in frames.items() if frame.empty]
        raise ValueError(f"CT-eligible splits are empty: {empty}")
    id_sets = {name: set(frame["patient_id"]) for name, frame in frames.items()}
    if any(
        id_sets[left] & id_sets[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise ValueError("Patient leakage detected between train/val/test")

    expected_metadata = build_normalization_metadata(
        frames["train"],
        args.split_file,
        args.ct_manifest,
        ct_config=ct_spec,
        ct_manifest=manifest,
        longitudinal_path=args.input,
    )
    stats = load_normalization_stats(args.stats_file)
    validate_normalization_metadata(stats, expected_metadata)

    record = REPRESENTATIONS[ct_spec.representation]
    usable = (
        manifest[record.status_column].astype(str).str.lower().isin(CT_SUCCESS_STATUSES)
    )
    cohort_paths = sorted(
        set(
            manifest.loc[
                usable & manifest["patient_id"].isin(table_ids), record.path_column
            ].astype(str)
        )
    )
    if not args.skip_array_content_check:
        for path in cohort_paths:
            load_ct_tensor(path, ct_spec.expected_shape, ct_spec.representation)

    split_summary = {
        name: {
            "patients": len(id_sets[name]),
            "observations": len(frame),
            "target_CysC_observed": int(frame["target_CysC_mask"].sum()),
            "target_CysC_missing": int((frame["target_CysC_mask"] == 0).sum()),
        }
        for name, frame in frames.items()
    }
    comparison_status = (
        table.drop_duplicates("patient_id")["comparison_status"]
        .fillna("<missing>")
        .astype(str)
        .value_counts()
        .to_dict()
        if "comparison_status" in table.columns
        else {"<column absent>": len(table_ids)}
    )
    report = {
        "status": "passed",
        "ct_input": ct_spec.to_dict(),
        "join": join_summary,
        "splits": split_summary,
        "comparison_status_patients": comparison_status,
        "ct_arrays_checked": 0 if args.skip_array_content_check else len(cohort_paths),
        "normalization_metadata": expected_metadata,
        "external_tools": {
            "totalsegmentator": shutil.which("totalsegmentator")
            or shutil.which("TotalSegmentator"),
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

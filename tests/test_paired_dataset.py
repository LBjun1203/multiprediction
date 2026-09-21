from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ct_preparation.manifest import CTInputSpec, canonicalize_manifest
from multidataset.huashan_dataset import ALL_BIOMARKERS
from pretraining.paired_dataset import (
    PairedCTBiochemistryDataset,
    build_paired_datasets,
    canonicalize_pair_table,
    fit_paired_normalization,
)


def pair_frame(patient_ids: list[str]) -> pd.DataFrame:
    rows = []
    for row_index, patient_id in enumerate(patient_ids):
        row: dict[str, object] = {"patient_id": patient_id}
        for marker_index, name in enumerate(ALL_BIOMARKERS):
            row[f"base_{name}"] = float(marker_index + row_index + 1)
        rows.append(row)
    return pd.DataFrame(rows)


class PairedDatasetTests(unittest.TestCase):
    def test_missing_values_masks_and_train_only_normalization(self) -> None:
        frame = pair_frame(["p1", "p2"])
        frame.loc[1, "base_CysC"] = np.nan
        frame = canonicalize_pair_table(frame)
        stats = fit_paired_normalization(frame.loc[frame["patient_id"].eq("p1")])
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "ct.npy"
            np.save(path, np.full((8, 8, 8), 0.5, np.float32))
            frame["ct_path_resolved"] = str(path)
            dataset = PairedCTBiochemistryDataset(
                frame,
                stats,
                CTInputSpec("roi_3d", "conv3d", (1, 8, 8, 8), "test-v1"),
            )
            sample = dataset[1]
            self.assertEqual(tuple(sample["ct"].shape), (1, 8, 8, 8))
            self.assertEqual(tuple(sample["bio"].shape), (len(ALL_BIOMARKERS),))
            self.assertEqual(float(sample["bio_mask"][-1]), 0.0)
            self.assertTrue(torch.isfinite(sample["bio"]).all())

    def test_duplicate_patient_and_forbidden_overlap_are_rejected(self) -> None:
        frame = pair_frame(["p1", "p1"])
        with self.assertRaisesRegex(ValueError, "one intended pair"):
            canonicalize_pair_table(frame)

        unique = canonicalize_pair_table(pair_frame(["p1", "p2"]))
        stats = fit_paired_normalization(unique)
        unique["ct_path_resolved"] = "missing.npy"
        with self.assertRaisesRegex(ValueError, "overlaps downstream"):
            PairedCTBiochemistryDataset(
                unique,
                stats,
                CTInputSpec("roi_3d", "conv3d", (1, 8, 8, 8), "test-v1"),
                forbidden_patient_ids={"p2"},
            )

    def test_build_bundle_excludes_downstream_validation_and_test(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            ids = ["p1", "p2", "p3", "p4", "p5"]
            table = pair_frame(ids)
            table_path = root / "pairs.csv"
            table.to_csv(table_path, index=False)
            ct_path = root / "ct.npy"
            np.save(ct_path, np.full((8, 8, 8), 0.25, np.float32))
            manifest = canonicalize_manifest(
                pd.DataFrame(
                    {
                        "patient_id": ids,
                        "roi_3d_path": [str(ct_path)] * len(ids),
                        "roi_3d_status": ["success"] * len(ids),
                        "roi_3d_preprocessing_version": ["test-v1"] * len(ids),
                    }
                )
            )
            manifest_path = root / "manifest.csv"
            manifest.to_csv(manifest_path, index=False)
            split_path = root / "pretrain_split.csv"
            pd.DataFrame(
                {"patient_id": ids, "split": ["train", "train", "val", "val", "train"]}
            ).to_csv(split_path, index=False)
            downstream_path = root / "downstream_split.csv"
            pd.DataFrame(
                {"patient_id": ["p5", "p9"], "split": ["test", "val"]}
            ).to_csv(downstream_path, index=False)
            bundle = build_paired_datasets(
                table_path,
                manifest_path,
                split_path,
                CTInputSpec("roi_3d", "resnet18_3d", (1, 8, 8, 8), "test-v1"),
                downstream_split_path=downstream_path,
            )
            self.assertEqual(len(bundle.train), 2)
            self.assertEqual(len(bundle.val), 2)
            self.assertEqual(
                bundle.preflight["excluded_downstream_val_test_patients"], 1
            )
            self.assertEqual(bundle.preflight["train_downstream_val_test_overlap"], 0)
            batch = next(iter(DataLoader(bundle.train, batch_size=2, shuffle=False)))
            self.assertEqual(tuple(batch["ct"].shape), (2, 1, 8, 8, 8))
            self.assertEqual(tuple(batch["bio"].shape), (2, len(ALL_BIOMARKERS)))


if __name__ == "__main__":
    unittest.main()

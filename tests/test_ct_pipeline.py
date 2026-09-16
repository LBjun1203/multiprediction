from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader

from ct_modal.config import load_experiment_config
from ct_modal.encoder_3d import CTEncoder3D
from ct_preparation.build_roi_3d import build_roi_3d_array
from ct_preparation.build_roi_25d import build_stack
from ct_preparation.manifest import (
    CTInputSpec,
    assert_same_ordered_cohort,
    canonicalize_manifest,
    representation_signature,
    resolve_ct_input_spec,
)
from evaluate import load_checkpoint_model
from multidataset.huashan_dataset import (
    ALL_BIOMARKERS,
    CORE_BIOMARKERS,
    HuashanMultimodalDataset,
    build_normalization_metadata,
    fit_normalization_stats,
    normalization_values_equal,
    save_normalization_stats,
    validate_normalization_metadata,
)
from multimodal_model import HuashanForecastModel
from preflight import main as preflight_main


def clinical_frame(
    path_column: str, path: Path, patient_ids: tuple[str, ...] = ("p1", "p2")
) -> pd.DataFrame:
    rows = []
    for index, patient_id in enumerate(patient_ids):
        row: dict[str, object] = {
            "patient_id": patient_id,
            "ct_path": "synthetic://ct",
            path_column: str(path),
            "age": 50.0 + index * 10,
            "sex_code": index % 2,
            "t_days": 100 + index,
            "base_CysC": 1.0 + index,
            "base_CysC_mask": 1,
            "target_CysC": 1.2 + index,
            "target_CysC_mask": 1,
        }
        for marker_index, name in enumerate(CORE_BIOMARKERS):
            row[f"base_{name}"] = float(marker_index + 1 + index)
            row[f"target_{name}"] = float(marker_index + 2 + index)
        rows.append(row)
    return pd.DataFrame(rows)


class ManifestAndConfigTests(unittest.TestCase):
    def test_versioned_experiment_configs_select_both_trainable_modes(self) -> None:
        baseline = load_experiment_config("configs/experiments/baseline_roi25d.yaml")
        volume = load_experiment_config("configs/experiments/ct_roi3d_seed42.yaml")
        self.assertEqual(baseline.ct.representation, "roi_25d")
        self.assertEqual(volume.ct.representation, "roi_3d")
        self.assertEqual(volume.ct.encoder, "conv3d")

    def test_legacy_aliases_and_unknown_columns_are_preserved(self) -> None:
        frame = pd.DataFrame(
            {
                "patient_id": ["1.0"],
                "nifti_path": ["ct.nii.gz"],
                "conversion_status": ["success"],
                "roi_path": ["roi.nii.gz"],
                "roi_status": ["success"],
                "L3_center_slice": [7],
                "npy_25d_path": ["roi.npy"],
                "build_25d_status": ["success"],
                "site_note": ["preserve me"],
            }
        )
        result = canonicalize_manifest(frame)
        self.assertEqual(result.loc[0, "patient_id"], "1")
        self.assertEqual(result.loc[0, "roi_25d_path"], "roi.npy")
        self.assertEqual(result.loc[0, "l3_center_index"], 7)
        self.assertEqual(result.loc[0, "site_note"], "preserve me")

    def test_incompatible_encoder_and_full_3d_fail_early(self) -> None:
        with self.assertRaisesRegex(ValueError, "incompatible"):
            resolve_ct_input_spec(
                {
                    "representation": "roi_3d",
                    "encoder": "resnet18_2d",
                    "expected_shape": [1, 16, 32, 32],
                    "preprocessing_version": "v1",
                }
            )
        with self.assertRaisesRegex(ValueError, "not registered as a trainable"):
            resolve_ct_input_spec(
                {
                    "representation": "full_3d",
                    "encoder": "none",
                    "expected_shape": [1, 1, 1, 1],
                    "preprocessing_version": "v1",
                }
            )


class PreparationTests(unittest.TestCase):
    def test_roi_25d_baseline_values(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "roi.nii.gz"
            volume = np.stack(
                [
                    np.full((4, 4), -200.0, np.float32),
                    np.full((4, 4), 50.0, np.float32),
                    np.full((4, 4), 300.0, np.float32),
                ]
            )
            sitk.WriteImage(sitk.GetImageFromArray(volume), str(path))
            result = build_stack(path, 1, 3, 4, -200.0, 300.0)
            expected = np.stack(
                [
                    np.zeros((4, 4), np.float32),
                    np.full((4, 4), 0.5, np.float32),
                    np.ones((4, 4), np.float32),
                ]
            )
            np.testing.assert_array_equal(result, expected)
            self.assertEqual(result.dtype, np.float32)

    def test_roi_3d_fixed_shape_air_padding_and_window(self) -> None:
        source = sitk.GetImageFromArray(np.zeros((4, 4, 4), dtype=np.float32))
        source.SetSpacing((2.0, 2.0, 2.0))
        array, cropped = build_roi_3d_array(
            source, (3.0, 3.0, 3.0), (1.0, 1.0, 1.0), (10, 10, 10), (-1024.0, 0.0)
        )
        self.assertEqual(array.shape, (10, 10, 10))
        self.assertEqual(array.dtype, np.float32)
        self.assertEqual(cropped.GetSpacing(), (1.0, 1.0, 1.0))
        self.assertEqual(float(array[0, 0, 0]), 0.0)
        self.assertEqual(float(array[5, 5, 5]), 1.0)


class DatasetModelTests(unittest.TestCase):
    def test_roi_3d_dataset_batch_and_encoder_backward(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "roi_3d.npy"
            np.save(path, np.full((8, 8, 8), 0.5, np.float32), allow_pickle=False)
            frame = clinical_frame("roi_3d_path", path)
            stats = fit_normalization_stats(frame)
            spec = CTInputSpec("roi_3d", "conv3d", (1, 8, 8, 8), "test-v1")
            dataset = HuashanMultimodalDataset(frame, stats, ct_config=spec)
            sample = dataset[0]
            self.assertEqual(tuple(sample["ct"].shape), (1, 8, 8, 8))
            self.assertEqual(sample["ct_representation"], "roi_3d")
            batch = next(iter(DataLoader(dataset, batch_size=2)))
            self.assertEqual(tuple(batch["ct"].shape), (2, 1, 8, 8, 8))
            encoder = CTEncoder3D(embedding_dim=16)
            output = encoder(batch["ct"])
            self.assertEqual(tuple(output.shape), (2, 16))
            output.sum().backward()
            self.assertTrue(
                any(parameter.grad is not None for parameter in encoder.parameters())
            )
            model = HuashanForecastModel(
                ct_embedding_dim=16,
                bio_embedding_dim=16,
                pretrained_ct=False,
                freeze_ct_backbone=False,
                ct_config=spec,
            )
            prediction = model(batch)
            self.assertEqual(tuple(prediction.shape), (2, len(ALL_BIOMARKERS)))
            prediction.sum().backward()
            model.eval()
            with torch.no_grad():
                reference = model(batch)
            checkpoint_path = Path(name) / "roi3d.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model.model_config,
                    "biomarker_order": ALL_BIOMARKERS,
                },
                checkpoint_path,
            )
            restored, _ = load_checkpoint_model(checkpoint_path, torch.device("cpu"))
            with torch.no_grad():
                actual = restored(batch)
            torch.testing.assert_close(actual, reference, rtol=0.0, atol=0.0)

    def test_legacy_checkpoint_defaults_to_roi_25d(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "old.pt"
            model = HuashanForecastModel(pretrained_ct=False)
            old_config = dict(model.model_config)
            old_config.pop("ct_config")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": old_config,
                    "biomarker_order": ALL_BIOMARKERS,
                },
                path,
            )
            restored, _ = load_checkpoint_model(path, torch.device("cpu"))
            self.assertEqual(restored.ct_input_spec.representation, "roi_25d")


class ProvenanceTests(unittest.TestCase):
    def test_unrelated_manifest_column_does_not_change_signature(self) -> None:
        spec = CTInputSpec("roi_25d", "resnet18_2d", (3, 32, 32), "v1")
        manifest = canonicalize_manifest(
            pd.DataFrame(
                {
                    "patient_id": ["p1"],
                    "roi_25d_path": ["a.npy"],
                    "roi_25d_status": ["success"],
                    "roi_25d_preprocessing_version": ["v1"],
                    "roi_3d_message": ["before"],
                }
            )
        )
        before = representation_signature(manifest, spec)
        manifest.loc[0, "roi_3d_message"] = "after"
        self.assertEqual(before, representation_signature(manifest, spec))
        manifest.loc[0, "roi_25d_status"] = "failed"
        self.assertNotEqual(before, representation_signature(manifest, spec))
        validate_normalization_metadata(
            {
                "metadata": {
                    "schema_version": 3,
                    "cohort_signature": "same",
                    "representation_signature": "old",
                }
            },
            {
                "schema_version": 3,
                "cohort_signature": "same",
                "representation_signature": "new",
            },
        )

    def test_effective_cohort_and_order_checks(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            split = Path(name) / "split.csv"
            split.write_text(
                "patient_id,split\np1,train\np2,val\np3,test\n", encoding="utf-8"
            )
            first = pd.DataFrame({"patient_id": ["p1"], "x": [1]})
            second = pd.DataFrame({"patient_id": ["p1", "p4"], "x": [1, 2]})
            one = build_normalization_metadata(first, split)
            two = build_normalization_metadata(second, split)
            self.assertNotEqual(one["cohort_signature"], two["cohort_signature"])
            self.assertTrue(
                normalization_values_equal(
                    {"base": {"x": 1}},
                    {"base": {"x": 1}, "metadata": {"schema_version": 3}},
                )
            )
        with self.assertRaisesRegex(ValueError, "Ordered comparison cohorts differ"):
            assert_same_ordered_cohort(["p1", "p2"], ["p2", "p1"])


class PreflightTests(unittest.TestCase):
    def test_preflight_checks_roi_25d_and_roi_3d(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            roi_25d = root / "roi_25d.npy"
            roi_3d = root / "roi_3d.npy"
            np.save(roi_25d, np.full((3, 32, 32), 0.5, np.float32))
            np.save(roi_3d, np.full((8, 8, 8), 0.5, np.float32))
            table = clinical_frame(
                "unused_path", roi_25d, patient_ids=("p1", "p2", "p3")
            )
            xlsx = root / "longitudinal.xlsx"
            table.to_excel(xlsx, sheet_name="model_long_0_2y", index=False)
            split = root / "split.csv"
            split.write_text(
                "patient_id,split\np1,train\np2,val\np3,test\n", encoding="utf-8"
            )
            stats_path = root / "normalization.json"
            save_normalization_stats(
                fit_normalization_stats(table.loc[table["patient_id"] == "p1"]),
                stats_path,
            )
            manifest = canonicalize_manifest(
                pd.DataFrame(
                    {
                        "patient_id": ["p1", "p2", "p3"],
                        "roi_25d_path": [str(roi_25d)] * 3,
                        "roi_25d_status": ["success"] * 3,
                        "roi_25d_preprocessing_version": ["test25d"] * 3,
                        "roi_3d_path": [str(roi_3d)] * 3,
                        "roi_3d_status": ["success"] * 3,
                        "roi_3d_preprocessing_version": ["test3d"] * 3,
                    }
                )
            )
            manifest_path = root / "manifest.csv"
            manifest.to_csv(manifest_path, index=False)
            configs = {
                "roi_25d": "representation: roi_25d\nencoder: resnet18_2d\nexpected_shape: [3, 32, 32]\npreprocessing_version: test25d\n",
                "roi_3d": "representation: roi_3d\nencoder: conv3d\nexpected_shape: [1, 8, 8, 8]\npreprocessing_version: test3d\n",
            }
            for representation, content in configs.items():
                config = root / f"{representation}.yaml"
                config.write_text(content, encoding="utf-8")
                argv = [
                    "preflight.py",
                    "--input",
                    str(xlsx),
                    "--ct-manifest",
                    str(manifest_path),
                    "--ct-config",
                    str(config),
                    "--split-file",
                    str(split),
                    "--stats-file",
                    str(stats_path),
                ]
                with patch("sys.argv", argv):
                    preflight_main()


if __name__ == "__main__":
    unittest.main()

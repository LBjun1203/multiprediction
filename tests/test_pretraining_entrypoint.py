from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from ct_preparation.manifest import canonicalize_manifest
from multidataset.huashan_dataset import ALL_BIOMARKERS
from pretrain_ct_bio import main as pretrain_main
from pretraining.checkpoint import load_checkpoint


class PretrainingEntrypointIntegrationTests(unittest.TestCase):
    def test_one_epoch_dataset_to_checkpoint_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            patient_ids = [f"p{index}" for index in range(6)]
            rows = []
            for patient_index, patient_id in enumerate(patient_ids):
                row: dict[str, object] = {"patient_id": patient_id}
                for marker_index, biomarker in enumerate(ALL_BIOMARKERS):
                    row[f"base_{biomarker}"] = float(patient_index + marker_index + 1)
                rows.append(row)
            pair_path = root / "pairs.csv"
            pd.DataFrame(rows).to_csv(pair_path, index=False)

            ct_path = root / "roi_3d.npy"
            np.save(ct_path, np.full((8, 16, 16), 0.5, dtype=np.float32))
            manifest = canonicalize_manifest(
                pd.DataFrame(
                    {
                        "patient_id": patient_ids,
                        "roi_3d_path": [str(ct_path)] * len(patient_ids),
                        "roi_3d_status": ["success"] * len(patient_ids),
                        "roi_3d_preprocessing_version": ["synthetic-v1"]
                        * len(patient_ids),
                    }
                )
            )
            manifest_path = root / "manifest.csv"
            manifest.to_csv(manifest_path, index=False)
            split_path = root / "split.csv"
            pd.DataFrame(
                {
                    "patient_id": patient_ids,
                    "split": ["train", "train", "train", "train", "val", "val"],
                }
            ).to_csv(split_path, index=False)
            ct_config_path = root / "ct.yaml"
            ct_config_path.write_text(
                "representation: roi_3d\n"
                "encoder: resnet18_3d\n"
                "expected_shape: [1, 8, 16, 16]\n"
                "preprocessing_version: synthetic-v1\n",
                encoding="utf-8",
            )
            config_path = root / "pretrain.yaml"
            config_path.write_text(
                "ct_config: ct.yaml\n"
                "objective: contrastive\n"
                "ct_feature_dim: 8\n"
                "ct_encoder_base_width: 2\n"
                "ct_dropout: 0.0\n"
                "bio_hidden_dim: 8\n"
                "bio_feature_dim: 8\n"
                "bio_dropout: 0.0\n"
                "projection_hidden_dim: 8\n"
                "projection_dim: 4\n"
                "temperature: 0.1\n"
                "learnable_temperature: false\n"
                "min_temperature: 0.01\n"
                "max_temperature: 1.0\n"
                "batch_size: 2\n"
                "epochs: 1\n"
                "ct_learning_rate: 0.001\n"
                "bio_learning_rate: 0.001\n"
                "head_learning_rate: 0.001\n"
                "weight_decay: 0.0\n"
                "amp: false\n"
                "gradient_clip: 1.0\n"
                "seed: 42\n"
                "early_stopping_metric: validation_loss\n"
                "patience: 1\n",
                encoding="utf-8",
            )
            output = root / "run"
            argv = [
                "pretrain_ct_bio.py",
                "--input",
                str(pair_path),
                "--ct-manifest",
                str(manifest_path),
                "--split-file",
                str(split_path),
                "--config",
                str(config_path),
                "--output-dir",
                str(output),
                "--device",
                "cpu",
            ]
            with patch("sys.argv", argv):
                pretrain_main()
            self.assertTrue((output / "best.pt").is_file())
            self.assertTrue((output / "last.pt").is_file())
            resolved = json.loads((output / "resolved_config.json").read_text())
            self.assertEqual(resolved["actual_batch_size"], 2)
            restored, checkpoint = load_checkpoint(output / "best.pt")
            self.assertEqual(checkpoint["epoch"], 1)
            restored.eval()
            with torch.no_grad():
                result = restored(
                    torch.rand(2, 1, 8, 16, 16),
                    torch.randn(2, 10),
                    torch.ones(2, 10),
                )
            self.assertEqual(tuple(result["z_ct"].shape), (2, 4))


if __name__ == "__main__":
    unittest.main()

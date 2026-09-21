from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from ct_preparation.manifest import CTInputSpec
from multimodal_model import HuashanForecastModel
from pretraining.checkpoint import (
    build_checkpoint,
    load_checkpoint,
    load_ct_encoder_for_forecasting,
    save_checkpoint,
)
from pretraining.contrastive_model import CTBioContrastiveModel
from pretraining.losses import retrieval_recalls, symmetric_info_nce


class ContrastivePretrainingTests(unittest.TestCase):
    def make_model(self) -> CTBioContrastiveModel:
        return CTBioContrastiveModel(
            CTInputSpec("roi_3d", "resnet18_3d", (1, 8, 16, 16), "test-v1"),
            ct_feature_dim=12,
            ct_encoder_base_width=2,
            ct_dropout=0.0,
            bio_hidden_dim=8,
            bio_feature_dim=12,
            bio_dropout=0.0,
            projection_hidden_dim=8,
            projection_dim=6,
            temperature=0.2,
        )

    def test_symmetric_loss_diagonal_labels_and_optimization(self) -> None:
        torch.manual_seed(5)
        model = self.make_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ct = torch.rand(3, 1, 8, 16, 16)
        bio = torch.randn(3, 10)
        mask = torch.ones(3, 10)
        before = model.ct_projector.layers[0].weight.detach().clone()
        output = model(ct, bio, mask)
        self.assertEqual(tuple(output["z_ct"].shape), (3, 6))
        torch.testing.assert_close(output["z_ct"].norm(dim=1), torch.ones(3))
        torch.testing.assert_close(output["z_bio"].norm(dim=1), torch.ones(3))
        loss, logits = symmetric_info_nce(
            output["z_ct"], output["z_bio"], output["temperature"]
        )
        labels = torch.arange(3)
        expected = (
            torch.nn.functional.cross_entropy(logits, labels)
            + torch.nn.functional.cross_entropy(logits.T, labels)
        ) / 2
        torch.testing.assert_close(loss, expected)
        self.assertEqual(tuple(logits.shape), (3, 3))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.ct_projector.layers[0].weight))
        metrics = retrieval_recalls(logits, (1, 5))
        self.assertIn("ct_to_bio_recall_at_1", metrics)

    def test_checkpoint_round_trip_restores_every_pretraining_module(self) -> None:
        torch.manual_seed(9)
        model = self.make_model().eval()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "pretrain.pt"
            checkpoint = build_checkpoint(
                model,
                optimizer,
                scaler,
                epoch=2,
                best_validation_metric=0.5,
                normalization_stats={"biomarker_order": []},
                split_provenance={"train_patient_count": 3},
                history=[{"epoch": 2}],
            )
            save_checkpoint(path, checkpoint)
            restored, loaded = load_checkpoint(path)
            restored.eval()
            self.assertEqual(loaded["checkpoint_type"], "ct_bio_contrastive_pretraining")
            for original, actual in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(actual, original, rtol=0, atol=0)

    def test_forecast_transfer_loads_only_ct_encoder(self) -> None:
        torch.manual_seed(11)
        pretraining_model = self.make_model()
        optimizer = torch.optim.AdamW(pretraining_model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "pretrain.pt"
            save_checkpoint(
                path,
                build_checkpoint(
                    pretraining_model,
                    optimizer,
                    scaler,
                    epoch=1,
                    best_validation_metric=1.0,
                    normalization_stats={"source": "synthetic"},
                    split_provenance={"source": "synthetic"},
                    history=[],
                ),
            )
            forecast = HuashanForecastModel(
                ct_embedding_dim=12,
                bio_embedding_dim=12,
                pretrained_ct=False,
                freeze_ct_backbone=False,
                ct_encoder_base_width=2,
                ct_config=pretraining_model.ct_input_spec,
            )
            bio_before = {
                key: value.detach().clone()
                for key, value in forecast.bio_encoder.state_dict().items()
            }
            fusion_before = {
                key: value.detach().clone()
                for key, value in forecast.fusion_head.state_dict().items()
            }
            provenance = load_ct_encoder_for_forecasting(
                forecast, path, policy="frozen"
            )
            self.assertEqual(provenance["transfer_policy"], "frozen")
            for key, value in pretraining_model.ct_encoder.state_dict().items():
                torch.testing.assert_close(
                    forecast.ct_encoder.state_dict()[key], value, rtol=0, atol=0
                )
            for key, value in bio_before.items():
                torch.testing.assert_close(
                    forecast.bio_encoder.state_dict()[key], value, rtol=0, atol=0
                )
            for key, value in fusion_before.items():
                torch.testing.assert_close(
                    forecast.fusion_head.state_dict()[key], value, rtol=0, atol=0
                )
            self.assertFalse(
                any(parameter.requires_grad for parameter in forecast.ct_encoder.parameters())
            )
            forecast_batch = {
                "ct": torch.rand(2, 1, 8, 16, 16),
                "bio_base": torch.randn(2, 10),
                "base_mask": torch.ones(2, 10),
                "demographics": torch.randn(2, 2),
                "time": torch.rand(2, 1),
            }
            prediction = forecast(forecast_batch)
            self.assertEqual(tuple(prediction.shape), (2, 10))
            prediction.square().mean().backward()
            self.assertTrue(
                any(
                    parameter.grad is not None
                    for parameter in forecast.fusion_head.parameters()
                )
            )
            checkpoint_keys = torch.load(path, weights_only=False).keys()
            self.assertIn("ct_projector_state", checkpoint_keys)
            self.assertNotIn("ct_projector", dict(forecast.named_modules()))

            incompatible = HuashanForecastModel(
                ct_embedding_dim=13,
                bio_embedding_dim=12,
                pretrained_ct=False,
                ct_encoder_base_width=2,
                ct_config=pretraining_model.ct_input_spec,
            )
            with self.assertRaisesRegex(ValueError, "feature dimension"):
                load_ct_encoder_for_forecasting(incompatible, path)


if __name__ == "__main__":
    unittest.main()

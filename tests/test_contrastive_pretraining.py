from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from ct_preparation.manifest import CTInputSpec
from pretraining.checkpoint import build_checkpoint, load_checkpoint, save_checkpoint
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


if __name__ == "__main__":
    unittest.main()

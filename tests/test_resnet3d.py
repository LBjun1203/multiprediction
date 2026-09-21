from __future__ import annotations

import unittest

import torch

from ct_modal.factory import build_ct_encoder
from ct_modal.resnet3d import ResNet18Encoder3D
from ct_preparation.manifest import CTInputSpec


class ResNet3DTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)

    def test_factory_forward_backward_and_save_load(self) -> None:
        spec = CTInputSpec("roi_3d", "resnet18_3d", (1, 16, 32, 32), "test-v1")
        encoder = build_ct_encoder(
            spec,
            embedding_dim=12,
            pretrained=False,
            dropout=0.0,
            freeze_backbone=False,
            base_width=4,
        )
        value = torch.rand(2, *spec.expected_shape)
        output = encoder(value)
        self.assertEqual(tuple(output.shape), (2, 12))
        output.square().mean().backward()
        self.assertTrue(
            any(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in encoder.parameters()
            )
        )

        restored = ResNet18Encoder3D(
            embedding_dim=12, dropout=0.0, base_width=4
        )
        restored.load_state_dict(encoder.state_dict(), strict=True)
        encoder.eval()
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(value), encoder(value), rtol=0, atol=0)

    def test_frozen_backbone_keeps_normalization_statistics_fixed(self) -> None:
        encoder = ResNet18Encoder3D(
            embedding_dim=8, dropout=0.0, freeze_backbone=True, base_width=4
        )
        running_mean = encoder.stem[1].running_mean.clone()
        encoder.train()
        encoder(torch.rand(2, 1, 16, 32, 32)).sum().backward()
        torch.testing.assert_close(encoder.stem[1].running_mean, running_mean)
        self.assertFalse(any(p.requires_grad for p in encoder.stem.parameters()))
        self.assertTrue(any(p.requires_grad for p in encoder.projection.parameters()))

    def test_input_validation(self) -> None:
        encoder = ResNet18Encoder3D(embedding_dim=8, base_width=4)
        valid = torch.rand(1, 1, 8, 16, 16)
        for invalid, message in (
            (torch.rand(2, 8, 16, 16), "one channel"),
            (torch.ones(1, 1, 8, 16, 16, dtype=torch.int64), "floating point"),
            (valid.clone().fill_(2.0), "range"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    encoder(invalid)
        non_finite = valid.clone()
        non_finite[0, 0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            encoder(non_finite)

    def test_production_shape_contract(self) -> None:
        encoder = ResNet18Encoder3D(
            embedding_dim=8, dropout=0.0, base_width=2
        ).eval()
        with torch.no_grad():
            output = encoder(torch.zeros(1, 1, 64, 160, 160))
        self.assertEqual(tuple(output.shape), (1, 8))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()

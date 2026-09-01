#!/usr/bin/env python3
"""Concat fusion and MLP time-query head for future biochemistry prediction."""

from __future__ import annotations

import argparse

import torch
from torch import nn
from torch.nn import functional as F

BIOMARKERS = [
    "HBA1C",
    "TG",
    "CHO",
    "HDL",
    "LDL",
    "ALB",
    "BUN",
    "UA",
    "CRE",
    "CysC",
]


class FusionTimeQueryHead(nn.Module):
    """Predict standardized future biomarkers from baseline embeddings and time.

    The module expects already encoded CT and baseline-biochemistry features.
    Demographics are ordered as ``[normalized_age, sex_code]`` and time is
    already expressed in years by the dataset (``t_days / 365.25``).
    """

    def __init__(
        self,
        ct_embedding_dim: int = 128,
        bio_embedding_dim: int = 128,
        demo_hidden_dim: int = 16,
        demo_embedding_dim: int = 32,
        fusion_hidden_dim: int = 128,
        fusion_dim: int = 128,
        time_hidden_dim: int = 32,
        time_embedding_dim: int = 32,
        decoder_hidden_dim: int = 128,
        decoder_bottleneck_dim: int = 64,
        num_outputs: int = len(BIOMARKERS),
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        dimensions = {
            "ct_embedding_dim": ct_embedding_dim,
            "bio_embedding_dim": bio_embedding_dim,
            "demo_hidden_dim": demo_hidden_dim,
            "demo_embedding_dim": demo_embedding_dim,
            "fusion_hidden_dim": fusion_hidden_dim,
            "fusion_dim": fusion_dim,
            "time_hidden_dim": time_hidden_dim,
            "time_embedding_dim": time_embedding_dim,
            "decoder_hidden_dim": decoder_hidden_dim,
            "decoder_bottleneck_dim": decoder_bottleneck_dim,
            "num_outputs": num_outputs,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if num_outputs != len(BIOMARKERS):
            raise ValueError(
                f"num_outputs must be {len(BIOMARKERS)} for the fixed biomarker "
                f"contract, got {num_outputs}"
            )
        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool):
            raise ValueError(  # noqa: TRY004 - public validation uses ValueError
                f"dropout must be a number in [0, 1], got {dropout!r}"
            )
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")

        self.ct_embedding_dim = ct_embedding_dim
        self.bio_embedding_dim = bio_embedding_dim
        self.demo_embedding_dim = demo_embedding_dim
        self.fusion_dim = fusion_dim
        self.time_embedding_dim = time_embedding_dim
        self.num_outputs = num_outputs

        self.demographics_encoder = nn.Sequential(
            nn.Linear(2, demo_hidden_dim),
            nn.GELU(),
            nn.Linear(demo_hidden_dim, demo_embedding_dim),
            nn.GELU(),
        )

        fusion_input_dim = ct_embedding_dim + bio_embedding_dim + demo_embedding_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_dim),
            nn.GELU(),
        )

        self.time_encoder = nn.Sequential(
            nn.Linear(1, time_hidden_dim),
            nn.GELU(),
            nn.Linear(time_hidden_dim, time_embedding_dim),
            nn.GELU(),
        )

        decoder_input_dim = fusion_dim + time_embedding_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, decoder_hidden_dim),
            nn.LayerNorm(decoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, decoder_bottleneck_dim),
            nn.GELU(),
            nn.Linear(decoder_bottleneck_dim, num_outputs),
        )

    def forward(
        self,
        ct_embedding: torch.Tensor,
        bio_embedding: torch.Tensor,
        demographics: torch.Tensor,
        time: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Return predictions, optionally together with intermediate features."""
        self._validate_inputs(ct_embedding, bio_embedding, demographics, time)

        demo_embedding = self.demographics_encoder(demographics)
        baseline_state = self.fusion(
            torch.cat([ct_embedding, bio_embedding, demo_embedding], dim=-1)
        )
        time_embedding = self.time_encoder(time)
        prediction = self.decoder(torch.cat([baseline_state, time_embedding], dim=-1))

        if return_features:
            return {
                "prediction": prediction,
                "baseline_state": baseline_state,
                "time_embedding": time_embedding,
                "demo_embedding": demo_embedding,
            }
        return prediction

    def _validate_inputs(
        self,
        ct_embedding: torch.Tensor,
        bio_embedding: torch.Tensor,
        demographics: torch.Tensor,
        time: torch.Tensor,
    ) -> None:
        inputs = {
            "ct_embedding": (ct_embedding, self.ct_embedding_dim),
            "bio_embedding": (bio_embedding, self.bio_embedding_dim),
            "demographics": (demographics, 2),
            "time": (time, 1),
        }

        for name, (tensor, expected_dim) in inputs.items():
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(  # noqa: TRY004 - public validation uses ValueError
                    f"{name} must be a torch.Tensor, got {type(tensor).__name__}"
                )
            if tensor.ndim not in (1, 2):
                raise ValueError(
                    f"{name} must have shape [{expected_dim}] or "
                    f"[batch, {expected_dim}], got {tuple(tensor.shape)}"
                )
            if tensor.shape[-1] != expected_dim:
                raise ValueError(
                    f"{name} last dimension must be {expected_dim}, "
                    f"got {tensor.shape[-1]}"
                )
            if not torch.is_floating_point(tensor):
                raise ValueError(f"{name} must be floating point, got {tensor.dtype}")
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f"{name} must contain only finite values")

        reference_shape = ct_embedding.shape[:-1]
        for name, (tensor, _) in inputs.items():
            if tensor.shape[:-1] != reference_shape:
                raise ValueError(
                    "all inputs must have matching batch dimensions; "
                    f"ct_embedding has {tuple(reference_shape)}, "
                    f"but {name} has {tuple(tensor.shape[:-1])}"
                )
            if tensor.device != ct_embedding.device:
                raise ValueError(
                    "all inputs must be on the same device; "
                    f"ct_embedding is on {ct_embedding.device}, "
                    f"but {name} is on {tensor.device}"
                )
            if tensor.dtype != ct_embedding.dtype:
                raise ValueError(
                    "all inputs must have the same dtype; "
                    f"ct_embedding is {ct_embedding.dtype}, "
                    f"but {name} is {tensor.dtype}"
                )


def masked_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """Return mean elementwise Huber loss over supervised target entries."""
    tensors = {
        "prediction": prediction,
        "target": target,
        "target_mask": target_mask,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(  # noqa: TRY004 - public validation uses ValueError
                f"{name} must be a torch.Tensor, got {type(tensor).__name__}"
            )
        if tensor.ndim != 2 or tensor.shape[-1] != len(BIOMARKERS):
            raise ValueError(
                f"{name} must have shape [batch, {len(BIOMARKERS)}], "
                f"got {tuple(tensor.shape)}"
            )
        if not torch.is_floating_point(tensor):
            raise ValueError(f"{name} must be floating point, got {tensor.dtype}")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name} must contain only finite values")

    if prediction.shape != target.shape or prediction.shape != target_mask.shape:
        raise ValueError(
            "prediction, target, and target_mask must have identical shapes, got "
            f"{tuple(prediction.shape)}, {tuple(target.shape)}, and "
            f"{tuple(target_mask.shape)}"
        )
    if prediction.device != target.device or prediction.device != target_mask.device:
        raise ValueError(
            "prediction, target, and target_mask must be on the same device"
        )
    if prediction.dtype != target.dtype or prediction.dtype != target_mask.dtype:
        raise ValueError("prediction, target, and target_mask must have the same dtype")
    if not torch.all((target_mask == 0) | (target_mask == 1)).item():
        raise ValueError("target_mask must contain only binary values 0 or 1")
    if not isinstance(delta, (int, float)) or isinstance(delta, bool) or delta <= 0:
        raise ValueError(f"delta must be a positive number, got {delta!r}")

    valid_count = target_mask.sum()
    if valid_count.item() <= 0:
        raise ValueError("target_mask must contain at least one supervised entry")

    elementwise_loss = F.huber_loss(
        prediction,
        target,
        reduction="none",
        delta=float(delta),
    )
    return (elementwise_loss * target_mask).sum() / valid_count


def _run_smoke_test() -> None:
    torch.manual_seed(42)
    model = FusionTimeQueryHead()
    model.eval()

    batch_size = 4
    ct = torch.randn(batch_size, 128)
    bio = torch.randn(batch_size, 128)
    demo = torch.randn(batch_size, 2)
    time = torch.rand(batch_size, 1) * 2.0

    with torch.no_grad():
        features = model(ct, bio, demo, time, return_features=True)
        prediction = features["prediction"]
        single_prediction = model(ct[0], bio[0], demo[0], time[0])

    assert prediction.shape == (batch_size, len(BIOMARKERS))
    assert torch.isfinite(prediction).all()
    assert features["baseline_state"].shape == (batch_size, 128)
    assert features["time_embedding"].shape == (batch_size, 32)
    assert features["demo_embedding"].shape == (batch_size, 32)
    assert single_prediction.shape == (len(BIOMARKERS),)
    assert torch.isfinite(single_prediction).all()

    target = torch.randn(batch_size, len(BIOMARKERS))
    mask = torch.ones(batch_size, len(BIOMARKERS))
    mask[0, 9] = 0
    mask[2, 9] = 0
    loss = masked_huber_loss(prediction, target, mask)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    changed_target = target.clone()
    changed_target[mask == 0] = 1_000_000.0
    changed_loss = masked_huber_loss(prediction, changed_target, mask)
    assert torch.allclose(loss, changed_loss, rtol=0.0, atol=1e-7)

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"biomarker order:             {BIOMARKERS}")
    print(f"batch prediction shape:      {list(prediction.shape)}")
    print(f"single-sample shape:         {list(single_prediction.shape)}")
    print(f"demo embedding shape:        {list(features['demo_embedding'].shape)}")
    print(f"baseline state shape:        {list(features['baseline_state'].shape)}")
    print(f"time embedding shape:        {list(features['time_embedding'].shape)}")
    print(f"trainable parameters:        {trainable_parameters}")
    print(f"masked Huber loss:           {loss.item():.8f} (passed)")
    print("masked CysC correctness:     passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run shape, finiteness, and masked-loss checks",
    )
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

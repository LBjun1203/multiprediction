#!/usr/bin/env python3
"""Biochemistry encoder for normalized baseline biomarkers and missingness masks."""

from __future__ import annotations

import argparse

import torch
from torch import nn


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


class BiochemistryEncoder(nn.Module):
    """Encode baseline biochemistry values and their observation masks.

    Inputs are expected to have shape ``[10]`` for one sample or ``[B, 10]``
    for a batch. Values must already be normalized and missing values must
    already be filled by the dataset layer. A mask value of 1 means observed;
    0 means missing.
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 64,
        embedding_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be positive, got {embedding_dim}"
            )
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

    def forward(
        self,
        bio_base: torch.Tensor,
        base_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the biochemistry embedding without altering the inputs."""
        self._validate_inputs(bio_base, base_mask)
        features = torch.cat([bio_base, base_mask], dim=-1)
        return self.encoder(features)

    def _validate_inputs(
        self,
        bio_base: torch.Tensor,
        base_mask: torch.Tensor,
    ) -> None:
        if not isinstance(bio_base, torch.Tensor):
            raise ValueError(
                f"bio_base must be a torch.Tensor, got {type(bio_base).__name__}"
            )
        if not isinstance(base_mask, torch.Tensor):
            raise ValueError(
                f"base_mask must be a torch.Tensor, got {type(base_mask).__name__}"
            )
        if bio_base.shape != base_mask.shape:
            raise ValueError(
                "bio_base and base_mask must have identical shapes, got "
                f"{tuple(bio_base.shape)} and {tuple(base_mask.shape)}"
            )
        if bio_base.ndim not in (1, 2):
            raise ValueError(
                "bio_base and base_mask must have shape [input_dim] or "
                f"[batch, input_dim], got {tuple(bio_base.shape)}"
            )
        if bio_base.shape[-1] != self.input_dim:
            raise ValueError(
                f"last input dimension must be {self.input_dim}, "
                f"got {bio_base.shape[-1]}"
            )
        if bio_base.device != base_mask.device:
            raise ValueError(
                "bio_base and base_mask must be on the same device, got "
                f"{bio_base.device} and {base_mask.device}"
            )
        if not torch.is_floating_point(bio_base):
            raise ValueError(f"bio_base must be floating point, got {bio_base.dtype}")
        if not torch.is_floating_point(base_mask):
            raise ValueError(f"base_mask must be floating point, got {base_mask.dtype}")
        if not torch.isfinite(bio_base).all().item():
            raise ValueError("bio_base must contain only finite values")
        if not torch.isfinite(base_mask).all().item():
            raise ValueError("base_mask must contain only finite values")
        if not torch.all((base_mask == 0) | (base_mask == 1)).item():
            raise ValueError("base_mask must contain only binary values 0 or 1")


def _run_smoke_test() -> None:
    torch.manual_seed(42)
    model = BiochemistryEncoder()

    bio = torch.randn(4, len(BIOMARKERS))
    mask = torch.ones(4, len(BIOMARKERS))
    mask[0, 9] = 0
    mask[2, 9] = 0
    output = model(bio, mask)

    assert output.shape == (4, 128)
    assert torch.isfinite(output).all()

    single_output = model(bio[0], mask[0])
    assert single_output.shape == (128,)
    assert torch.isfinite(single_output).all()

    invalid_mask = mask.clone()
    invalid_mask[0, 9] = 0.5
    try:
        model(bio, invalid_mask)
    except ValueError as error:
        assert "0 or 1" in str(error)
    else:
        raise AssertionError("non-binary base_mask was not rejected")

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"input bio shape:        {list(bio.shape)}")
    print(f"input mask shape:       {list(mask.shape)}")
    print(f"output embedding shape: {list(output.shape)}")
    print(f"single-sample shape:    {list(single_output.shape)}")
    print(f"trainable parameters:   {trainable_parameters}")
    print("mask validation:        passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run batch, single-sample, and mask-validation checks",
    )
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

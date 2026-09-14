#!/usr/bin/env python3
"""Fast deterministic checks for the add-only comparison experiment layer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import masked_huber_numpy
from experiments.evaluate_classical_baselines import (
    fit_ridge,
    predict_ridge,
)
from experiments.tabular_model import TabularForecastModel


def test_metrics() -> None:
    target = np.arange(30, dtype=np.float64).reshape(3, 10) / 10.0
    mask = np.ones_like(target)
    mask[0, 9] = 0
    prediction = target.copy()
    prediction[0, 9] = 1_000_000.0
    assert masked_huber_numpy(prediction, target, mask) == 0.0


def test_ridge() -> None:
    rng = np.random.default_rng(42)
    features = rng.normal(size=(100, 4))
    truth = np.asarray([0.4, -1.2, 0.7, 2.0])
    target = 1.5 + features @ truth
    coefficients = fit_ridge(
        features, target, np.ones(len(features), dtype=np.float64), alpha=0.0
    )
    prediction = predict_ridge(features, coefficients)
    assert np.max(np.abs(prediction - target)) < 1e-9


def test_tabular_network() -> None:
    torch.manual_seed(42)
    model = TabularForecastModel(dropout=0.0)
    batch = {
        "bio_base": torch.randn(5, 10),
        "base_mask": torch.ones(5, 10),
        "demographics": torch.randn(5, 2),
        "time": torch.rand(5, 1) * 2.0,
    }
    batch["base_mask"][0, 9] = 0
    output = model(batch)
    assert output.shape == (5, 10)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def main() -> None:
    test_metrics()
    test_ridge()
    test_tabular_network()
    print("experiments smoke test: passed")


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from maxwell_environment_optimizer import MaxwellEnvironmentOptimizer


def test_environment_optimization_reduces_env():
    baseline = torch.tensor([0.6, 0.7, 0.5, 0.8, 0.5, 0.9])
    optimizer = MaxwellEnvironmentOptimizer(iterations=200)
    result = optimizer.optimize(baseline)
    assert result.transformed_pestle.shape == (6,)
    assert result.transform_matrix.shape == (6, 6)
    assert result.transformed_pestle[5] <= baseline[5]

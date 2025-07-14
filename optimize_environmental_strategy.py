"""Demonstration script for MaxwellEnvironmentOptimizer."""

import torch
from maxwell_environment_optimizer import MaxwellEnvironmentOptimizer

# Example baseline PESTLE vector for a single company
# [Political, Economic, Social, Technological, Legal, Environmental]
BASELINE_PESTLE = torch.tensor([0.6, 0.7, 0.5, 0.8, 0.5, 0.9])

if __name__ == "__main__":
    optimizer = MaxwellEnvironmentOptimizer()
    result = optimizer.optimize(BASELINE_PESTLE)
    print("Transformed PESTLE:", result.transformed_pestle.numpy())
    print("Impact percentages:", result.impact_percentages.numpy())
    print("Transformation matrix:\n", result.transform_matrix.numpy())

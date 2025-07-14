import torch
from dataclasses import dataclass
from typing import Tuple

@dataclass
class OptimizationResult:
    transformed_pestle: torch.Tensor
    impact_percentages: torch.Tensor
    transform_matrix: torch.Tensor

class MaxwellEnvironmentOptimizer:
    """Optimize a company's PESTLE impact to minimize environmental score."""

    def __init__(self, lr: float = 0.05, iterations: int = 300):
        self.lr = lr
        self.iterations = iterations
        self.transform = torch.eye(6, requires_grad=True)

    def optimize(self, pestle_vector: torch.Tensor) -> OptimizationResult:
        pestle_vector = pestle_vector.detach()
        optimizer = torch.optim.Adam([self.transform], lr=self.lr)
        baseline = pestle_vector.clone()

        for _ in range(self.iterations):
            optimizer.zero_grad()
            projected = baseline @ self.transform
            # Loss: minimize environmental factor with small deviation on others
            env_loss = projected[5]
            other_loss = torch.sum((projected[:5] - baseline[:5]) ** 2)
            loss = env_loss + 0.1 * other_loss
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            final_pestle = baseline @ self.transform
            diff = final_pestle - baseline
            impact_percent = diff / (baseline + 1e-8) * 100
        return OptimizationResult(final_pestle, impact_percent, self.transform.detach())

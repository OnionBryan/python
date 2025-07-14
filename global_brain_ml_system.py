from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class GBNode:
    """Minimal node holding features and PESTLE factors."""

    features: torch.Tensor
    pestle_factors: torch.Tensor

    def get_state_vector(self) -> torch.Tensor:
        return self.features


class GlobalBrainMLSystem:
    """Simple system that predicts PESTLE factors for a set of nodes."""

    def __init__(self, num_nodes: int, features_per_node: int) -> None:
        self.is_active = False
        self.epochs = 0
        self.loss = 0.0
        self.num_nodes = num_nodes

        input_size = num_nodes * features_per_node
        output_size = num_nodes * 6

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_size),
            torch.nn.Sigmoid(),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def gather_system_state(self, nodes: List[GBNode]) -> torch.Tensor:
        """Concatenate node state vectors into a single batch."""
        return torch.cat([n.get_state_vector() for n in nodes]).unsqueeze(0)

    def run_training_cycle(
        self, nodes: List[GBNode], entropy: float, coherence: float
    ) -> None:
        if not self.is_active:
            return

        system_state = self.gather_system_state(nodes)
        predicted = self.model(system_state)
        predicted_matrix = predicted.view(self.num_nodes, 6)

        coherence_loss = 1.0 - coherence
        entropy_loss = entropy
        loss = predicted_matrix.sum() * 0.0 + (coherence_loss + entropy_loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            for idx, node in enumerate(nodes):
                node.pestle_factors += 0.1 * (
                    predicted_matrix[idx] - node.pestle_factors
                )

        self.loss = float(loss)
        self.epochs += 1

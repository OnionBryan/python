from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable

import matplotlib.pyplot as plt
import networkx as nx
import torch


@dataclass
class Node:
    index: int
    name: str
    features: torch.Tensor
    prestige: float = 0.0
    connectivity: float = 0.0


class NetworkSimulation:
    """A very small network simulation using PyTorch tensors."""

    def __init__(self, num_nodes: int = 50, feature_dim: int = 10, device_to_use: str | torch.device = "cpu"):
        self.device = torch.device(device_to_use)
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.params: Dict[str, float] = {
            "competition": 0.1,
            "feature_influence": 0.5,
            "learning_rate": 0.01,
        }
        self.nodes: List[Node] = []
        self.connections: Dict[Tuple[int, int], Dict[str, float]] = {}
        self.interaction_matrix = torch.zeros(feature_dim, feature_dim, device=self.device)
        self.components: List[Iterable[int]] = []

    # ------------------------------------------------------------------
    def init(self) -> "NetworkSimulation":
        """Initialize the nodes and connections."""
        feature_matrix = torch.randn(self.num_nodes, self.feature_dim, device=self.device)
        self.nodes = [Node(i, f"Node {i}", feature_matrix[i]) for i in range(self.num_nodes)]
        self.update_connections()
        self.calculate_components()
        return self

    # ------------------------------------------------------------------
    def update(self, dt: float = 1.0) -> None:
        """Update node features based on connection influences."""
        feature_matrix = torch.stack([n.features for n in self.nodes])

        conn_matrix = torch.zeros(self.num_nodes, self.num_nodes, device=self.device)
        for (a, b), conn in self.connections.items():
            conn_matrix[a, b] = conn["score"]
            conn_matrix[b, a] = conn["score"]

        influence = self.params["feature_influence"]
        learning = self.params["learning_rate"]

        effect = conn_matrix @ feature_matrix * influence / max(1, self.num_nodes)
        feature_matrix = feature_matrix + learning * effect * dt

        for idx, node in enumerate(self.nodes):
            node.features = feature_matrix[idx]
            node.connectivity = torch.count_nonzero(conn_matrix[idx]).item()

        self.interaction_matrix = feature_matrix.T @ feature_matrix / max(1, self.num_nodes)

    # ------------------------------------------------------------------
    def update_connections(self) -> None:
        """Recompute connection scores based on feature similarity."""
        feature_matrix = torch.stack([n.features for n in self.nodes])
        similarity = feature_matrix @ feature_matrix.T
        similarity.fill_diagonal_(0)
        scores = torch.sigmoid(similarity - self.params["competition"])

        self.connections.clear()
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                score = scores[i, j].item()
                if score > 0.1:
                    self.connections[(i, j)] = {
                        "a": i,
                        "b": j,
                        "score": score,
                        "alpha": score * 10 + 1,
                        "beta": (1 - score) * 10 + 1,
                    }

    # ------------------------------------------------------------------
    def calculate_components(self) -> None:
        """Determine connected components and update node prestige."""
        graph = nx.Graph()
        graph.add_nodes_from(range(self.num_nodes))
        graph.add_edges_from(self.connections.keys())

        self.components = list(nx.connected_components(graph))

        for comp in self.components:
            prestige = len(comp) / self.num_nodes
            for idx in comp:
                self.nodes[idx].prestige = prestige

    # ------------------------------------------------------------------
    def bayesian_update(self, observations: List[Tuple[int, int, float]]) -> None:
        """Update connection scores from observed outcomes."""
        for a, b, outcome in observations:
            key = (min(a, b), max(a, b))
            conn = self.connections.get(key)
            if conn is None:
                continue
            conn["alpha"] += outcome
            conn["beta"] += 1 - outcome
            conn["score"] = conn["alpha"] / (conn["alpha"] + conn["beta"])

    # ------------------------------------------------------------------
    def visualize_network(self) -> None:
        """Draw the current network using networkx."""
        graph = nx.Graph()
        graph.add_nodes_from(range(self.num_nodes))
        weights = []
        for (a, b), conn in self.connections.items():
            graph.add_edge(a, b, weight=conn["score"])
            weights.append(conn["score"])
        pos = nx.spring_layout(graph, seed=42)
        plt.figure(figsize=(8, 6))
        nx.draw(
            graph,
            pos,
            with_labels=True,
            node_color="lightblue",
            width=[w * 2 for w in weights],
            edge_color=weights,
            edge_cmap=plt.cm.Blues,
        )
        plt.title("Network Connectivity")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    def visualize_feature_space(self, dimensions: Tuple[int, int] = (0, 1)) -> None:
        """Scatter plot of nodes in the selected feature dimensions."""
        x_dim, y_dim = dimensions
        feature_matrix = torch.stack([n.features for n in self.nodes]).cpu()
        plt.figure(figsize=(6, 5))
        plt.scatter(feature_matrix[:, x_dim], feature_matrix[:, y_dim], c="orange")
        for node in self.nodes:
            x, y = node.features[x_dim].item(), node.features[y_dim].item()
            plt.text(x, y, node.name, fontsize=8)
        plt.xlabel(f"Feature {x_dim}")
        plt.ylabel(f"Feature {y_dim}")
        plt.title("Feature Space")
        plt.tight_layout()
        plt.show()


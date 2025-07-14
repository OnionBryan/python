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


class HilbertCurve:
    """Simple 2D Hilbert curve encoder/decoder."""

    def __init__(self, order: int = 3) -> None:
        self.order = order
        self.size = 1 << order

    def index_to_coord(self, index: int) -> Tuple[int, int]:
        x = y = 0
        n = 1
        for s in range(self.order):
            rx = 1 & (index >> 1)
            ry = 1 & (index ^ rx)
            if ry == 0:
                if rx == 1:
                    x, y = self.size - 1 - x, self.size - 1 - y
                x, y = y, x
            x += n * rx
            y += n * ry
            index >>= 2
            n <<= 1
        return x, y


class MatrixXORNet:
    """Tiny XOR neural network using matrix ops."""

    def __init__(self, lr: float = 0.1, device: torch.device | str = "cpu") -> None:
        device = torch.device(device)
        self.W1 = torch.randn(2, 2, device=device)
        self.b1 = torch.zeros(2, device=device)
        self.W2 = torch.randn(2, 1, device=device)
        self.b2 = torch.zeros(1, device=device)
        self.lr = lr
        self.device = device

    def _forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.sigmoid(x @ self.W1 + self.b1)
        o = torch.sigmoid(h @ self.W2 + self.b2)
        return o, h

    def train(self, epochs: int = 2000) -> float:
        x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], device=self.device)
        y = torch.tensor([[0.], [1.], [1.], [0.]], device=self.device)
        for _ in range(epochs):
            o, h = self._forward(x)
            grad_o = 2 * (o - y) * o * (1 - o)
            grad_W2 = h.t() @ grad_o
            grad_b2 = grad_o.sum(0)
            grad_h = grad_o @ self.W2.t()
            grad_W1 = x.t() @ (grad_h * h * (1 - h))
            grad_b1 = (grad_h * h * (1 - h)).sum(0)
            self.W2 -= self.lr * grad_W2
            self.b2 -= self.lr * grad_b2
            self.W1 -= self.lr * grad_W1
            self.b1 -= self.lr * grad_b1
        loss = ((self._forward(x)[0] - y) ** 2).mean()
        return loss.item()

    def accuracy(self) -> float:
        x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], device=self.device)
        y = torch.tensor([[0.], [1.], [1.], [0.]], device=self.device)
        pred = (self._forward(x)[0] > 0.5).float()
        return (pred == y).float().mean().item()


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
        self.hilbert = HilbertCurve()
        self.xor_net = MatrixXORNet(device=self.device)

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

    # ------------------------------------------------------------------
    def hilbert_positions(self) -> List[Tuple[int, int]]:
        """Return Hilbert coordinates for each node index."""
        return [self.hilbert.index_to_coord(i) for i in range(self.num_nodes)]

    # ------------------------------------------------------------------
    def run_xor_training(self, epochs: int = 2000) -> float:
        """Train XOR network and return final loss."""
        return self.xor_net.train(epochs)

    def xor_accuracy(self) -> float:
        return self.xor_net.accuracy()


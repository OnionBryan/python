import torch
import matplotlib.pyplot as plt
import networkx as nx
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class Node:
    name: str
    features: torch.Tensor
    prestige: float = 0.0
    connectivity: float = 0.0


class NetworkSimulation:
    def __init__(self, num_nodes: int = 50, feature_dim: int = 10, device_to_use=None):
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.device = device_to_use if device_to_use is not None else torch.device('cpu')
        self.params: Dict[str, float] = {
            'competition': 0.1,
            'feature_influence': 0.5,
            'learning_rate': 0.01,
        }
        self.nodes: List[Node] = []
        self.connections: Dict[str, Dict[str, float]] = {}
        self.components: List[set] = []
        self.interaction_matrix: torch.Tensor = torch.randn(
            feature_dim, feature_dim, device=self.device)

    def init(self):
        self.nodes = [
            Node(
                name=f"Node{i}",
                features=torch.randn(self.feature_dim, device=self.device),
                prestige=torch.rand(1).item(),
            )
            for i in range(self.num_nodes)
        ]
        self.update_connections()
        self.calculate_components()
        return self

    def _feature_matrix(self) -> torch.Tensor:
        return torch.stack([n.features for n in self.nodes])

    def update(self, dt: float):
        with torch.no_grad():
            features = self._feature_matrix()
            # Matrix-based influence update
            influence = features @ self.interaction_matrix
            features = features + dt * influence
            for i, node in enumerate(self.nodes):
                node.features = features[i]
        return self

    def update_connections(self):
        features = self._feature_matrix()
        norms = features.norm(dim=1, keepdim=True) + 1e-8
        normalized = features / norms
        similarity = normalized @ normalized.T
        self.connections = {}
        for i in range(self.num_nodes):
            self.nodes[i].connectivity = similarity[i].mean().item()
            for j in range(i + 1, self.num_nodes):
                score = similarity[i, j].item()
                self.connections[f"{i}-{j}"] = {
                    'a': i,
                    'b': j,
                    'score': score,
                }
        return self

    def calculate_components(self, threshold: float = 0.5):
        G = nx.Graph()
        G.add_nodes_from(range(self.num_nodes))
        for key, conn in self.connections.items():
            if conn['score'] > threshold:
                G.add_edge(conn['a'], conn['b'])
        self.components = list(nx.connected_components(G))
        return self.components

    def bayesian_update(self, observations: List[Tuple[int, int, float]]):
        lr = self.params.get('learning_rate', 0.01)
        for a_idx, b_idx, outcome in observations:
            node_a = self.nodes[a_idx]
            node_b = self.nodes[b_idx]
            pred = torch.dot(node_a.features, node_b.features)
            error = outcome - pred.item()
            grad_a = error * node_b.features
            grad_b = error * node_a.features
            node_a.features = node_a.features + lr * grad_a
            node_b.features = node_b.features + lr * grad_b
        self.update_connections()

    def visualize_network(self):
        G = nx.Graph()
        for i in range(self.num_nodes):
            G.add_node(i)
        for key, conn in self.connections.items():
            if conn['score'] > 0.5:
                G.add_edge(conn['a'], conn['b'], weight=conn['score'])
        pos = {i: self.nodes[i].features[:2].cpu().numpy() for i in range(self.num_nodes)}
        weights = [d['weight'] for u, v, d in G.edges(data=True)]
        nx.draw(G, pos, with_labels=True, width=weights, node_color='skyblue')
        plt.show()

    def visualize_feature_space(self, dimensions: Tuple[int, int] = (0, 1)):
        dim_x, dim_y = dimensions
        data = self._feature_matrix().cpu()
        plt.figure()
        plt.scatter(data[:, dim_x], data[:, dim_y], c='orange')
        for i, node in enumerate(self.nodes):
            plt.text(data[i, dim_x].item(), data[i, dim_y].item(), node.name)
        plt.xlabel(f'Feature {dim_x}')
        plt.ylabel(f'Feature {dim_y}')
        plt.tight_layout()
        plt.show()

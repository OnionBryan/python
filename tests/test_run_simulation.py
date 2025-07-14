import importlib
import sys
import types
import numpy as np


def create_stubs(cuda_available=False):
    # Torch stub
    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)

    class Device:
        def __init__(self, name):
            self.name = name
        def __str__(self):
            return self.name
    torch_stub.device = Device

    # Matplotlib stub
    plt_stub = types.ModuleType("matplotlib.pyplot")
    def noop(*args, **kwargs):
        pass
    for name in ["figure", "imshow", "colorbar", "title", "xlabel", "ylabel", "tight_layout", "show"]:
        setattr(plt_stub, name, noop)
    mpl_stub = types.ModuleType("matplotlib")
    mpl_stub.pyplot = plt_stub

    # NetworkSimulation stub
    ns_stub = types.ModuleType("network_simulator")

    class Node:
        def __init__(self, name):
            self.name = name
            self.prestige = 1.0
            self.connectivity = 1.0

    class InteractionMatrix:
        def cpu(self):
            return types.SimpleNamespace(numpy=lambda: np.zeros((10, 10)))

    class NetworkSimulation:
        instances = []
        def __init__(self, device_to_use=None):
            self.device_to_use = device_to_use
            self.params = {}
            self.connections = {0: {'a': Node('A'), 'b': Node('B'), 'score': 1.0}}
            self.nodes = [Node('A'), Node('B')]
            self.interaction_matrix = InteractionMatrix()
            NetworkSimulation.instances.append(self)
        def init(self):
            return self
        def update(self, dt):
            pass
        def update_connections(self):
            pass
        def calculate_components(self):
            pass
        def bayesian_update(self, obs):
            self.last_observations = obs
        def visualize_network(self):
            pass
        def visualize_feature_space(self, dimensions):
            pass

    ns_stub.NetworkSimulation = NetworkSimulation

    return torch_stub, mpl_stub, plt_stub, ns_stub


def test_main_cpu(monkeypatch, capsys):
    torch_stub, mpl_stub, plt_stub, ns_stub = create_stubs(cuda_available=False)
    monkeypatch.setitem(sys.modules, 'torch', torch_stub)
    monkeypatch.setitem(sys.modules, 'matplotlib', mpl_stub)
    monkeypatch.setitem(sys.modules, 'matplotlib.pyplot', plt_stub)
    monkeypatch.setitem(sys.modules, 'network_simulator', ns_stub)

    run_simulation = importlib.import_module('run_simulation')
    importlib.reload(run_simulation)

    run_simulation.main()

    captured = capsys.readouterr()
    assert "Using device: cpu" in captured.out
    sim = ns_stub.NetworkSimulation.instances[-1]
    assert sim.params['competition'] == 0.15
    assert sim.params['feature_influence'] == 0.6
    assert sim.params['learning_rate'] == 0.02
    assert hasattr(sim, 'last_observations')

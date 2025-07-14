import sys
from pathlib import Path
import torch

# Ensure repository root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from global_brain_ml_system import GlobalBrainMLSystem, GBNode


def test_gather_system_state_shape():
    nodes = [GBNode(torch.ones(3), torch.zeros(6)) for _ in range(2)]
    system = GlobalBrainMLSystem(num_nodes=2, features_per_node=3)
    state = system.gather_system_state(nodes)
    assert state.shape == (1, 6)


def test_run_training_cycle_updates_pestle():
    nodes = [GBNode(torch.ones(2), torch.zeros(6)) for _ in range(2)]
    system = GlobalBrainMLSystem(num_nodes=2, features_per_node=2)
    system.is_active = True
    system.run_training_cycle(nodes, entropy=0.2, coherence=0.8)
    assert system.epochs == 1
    for node in nodes:
        assert not torch.allclose(node.pestle_factors, torch.zeros(6))

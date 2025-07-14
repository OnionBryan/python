# python
work in python

## Network Simulation

Run `python run_simulation.py` to execute the demonstration. The program
automatically uses CUDA when available. The main simulation logic lives in
`network_simulator.py` and can be initialized with a specific device:

```python
import torch
from network_simulator import NetworkSimulation

sim = NetworkSimulation(device_to_use=torch.device('cuda')).init()
```

Adjust parameters via `sim.params` before running update steps.

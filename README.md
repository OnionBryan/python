# python

A collection of experiments and scripts.

## Running the Network Simulation

1. Install the required packages:
   ```bash
   pip install torch matplotlib networkx
   ```
2. Execute the simulation:
   ```bash
   python run_simulation.py
   ```

The script will initialize a small network and display several plots demonstrating its dynamics.

## TensorFlow Setup

The repository includes a small utility to verify that TensorFlow is available.
Run the script to perform a simple matrix multiplication:

```bash
python tensorflow_setup.py
```

The output should be a 2x1 matrix computed by TensorFlow on the CPU.

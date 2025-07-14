# Python Projects

This repository hosts small experiments written in Python. The two main scripts
focus on network simulation and algorithmic music generation.

## Project Goals

* **Network simulation:** explore graph dynamics with a custom `NetworkSimulation`
  class (provided separately) and visualize interactions using PyTorch and
  Matplotlib.
* **Generative music:** create lofi-style ambient tracks entirely in Python using
  NumPy, SciPy and the Pedalboard effects library.

## Scripts

### `run_simulation.py`
Simulates a dynamic network. CUDA acceleration is used when available and the
results can be visualized through Matplotlib. A separate
`network_simulator` module is required for this script.

### `music.py`
Generates a complete lofi composition by building melodies, basslines, chords
and drum patterns. The final audio is processed with effects such as a compressor,
low-pass filter and reverb to create a finished WAV file.

## Dependencies

External Python packages used in this repository:

- `numpy`
- `scipy`
- `torch`
- `matplotlib`
- `pedalboard`
- `network_simulator` (external module needed for the simulation script)

## Example Usage

Run the network simulation:

```bash
python run_simulation.py
```

Generate a lofi track:

```bash
python music.py
```

import time
import torch
import matplotlib.pyplot as plt
from network_simulator import NetworkSimulation


def main():
    print("Network Simulation with CUDA Acceleration")
    print("=========================================")

    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"Using device: {device}")

    # Create and initialize the simulation
    print("\nInitializing simulation...")
    sim = NetworkSimulation(device_to_use=device).init()

    xor_loss = sim.run_xor_training(3000)
    print(f"XOR loss after training: {xor_loss:.4f}  (acc {sim.xor_accuracy()*100:.1f}%)")

    # Configure simulation parameters
    sim.params['competition'] = 0.15
    sim.params['feature_influence'] = 0.6
    sim.params['learning_rate'] = 0.02

    print("\nSimulation parameters:")
    for param, value in sim.params.items():
        print(f"  {param}: {value}")

    # Run the simulation
    print("\nRunning simulation...")
    start_time = time.time()

    # Run for 1000 steps
    steps = 1000
    for i in range(steps):
        sim.update(0.01)

        # Update connections every 100 steps
        if i % 100 == 0:
            sim.update_connections()
            sim.calculate_components()
            print(f"  Step {i}/{steps} completed")

            # Generate some observations for Bayesian learning
            if i > 0:
                # Create synthetic observations based on current node relationships
                observations = []
                for key, conn in list(sim.connections.items())[:5]:  # Use first 5 connections
                    node_a = conn['a']
                    node_b = conn['b']
                    # Use connection score as outcome measure
                    outcome = conn['score']
                    observations.append((node_a, node_b, outcome))

                # Apply Bayesian update
                sim.bayesian_update(observations)
                print(f"  Applied Bayesian update with {len(observations)} observations")

    elapsed_time = time.time() - start_time
    print(f"\nSimulation completed in {elapsed_time:.2f} seconds")

    # Print network statistics
    print("\nNetwork Statistics:")
    print(f"  Number of nodes: {len(sim.nodes)}")
    print(f"  Number of connections: {len(sim.connections)}")

    # Calculate average prestige
    avg_prestige = sum(node.prestige for node in sim.nodes) / len(sim.nodes)
    print(f"  Average prestige: {avg_prestige:.4f}")

    # Find node with highest connectivity
    most_connected = max(sim.nodes, key=lambda n: n.connectivity)
    print(f"  Most connected node: {most_connected.name} ({most_connected.connectivity:.2f})")

    print("\nHilbert coordinates (first 5 nodes):")
    for idx, coord in enumerate(sim.hilbert_positions()[:5]):
        print(f"  Node {idx}: {coord}")

    # Visualize the network
    print("\nVisualizing network...")
    sim.visualize_network()

    # Visualize different dimensions of the feature space
    print("\nVisualizing feature space...")
    for dims in [(0, 1), (2, 3), (4, 5)]:
        print(f"  Showing dimensions {dims}")
        sim.visualize_feature_space(dimensions=dims)

    # Visualize the interaction matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(sim.interaction_matrix.cpu().numpy(), cmap='viridis')
    plt.colorbar(label='Interaction Strength')
    plt.title('10x10 Interaction Matrix')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Feature Dimension')
    plt.tight_layout()
    plt.show()

    print("\nSimulation demonstration complete.")


if __name__ == "__main__":
    main()

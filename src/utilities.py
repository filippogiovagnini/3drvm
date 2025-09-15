import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(project_root)

import matplotlib.pyplot as plt


def plot_vectors(pos, vec, sample_rate=2, zoom_out_factor=1.2, time_steps = 3, nu = 0.01, epochs_simulation = 1000, realizations = 2, final_time = 0.1, label="Std"):
    """
    Plots particles at pos[i, j, k, :] with vectors vec[i, j, k, :].
    
    Args:
        pos: JAX array (N, N, N, 3) - particle positions
        vec: JAX array (N, N, N, 3) - attached vectors
        sample_rate: int - how many points to skip to avoid cluttering
        zoom_out_factor: float - expands the plot range (default 1.2 means 20% wider)

    Returns:
        None (saves the plot as an image file)
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Extract downsampled positions and vectors
    X = pos[::sample_rate, ::sample_rate, ::sample_rate, 0].flatten()
    Y = pos[::sample_rate, ::sample_rate, ::sample_rate, 1].flatten()
    Z = pos[::sample_rate, ::sample_rate, ::sample_rate, 2].flatten()

    U = vec[::sample_rate, ::sample_rate, ::sample_rate, 0].flatten()
    V = vec[::sample_rate, ::sample_rate, ::sample_rate, 1].flatten()
    W = vec[::sample_rate, ::sample_rate, ::sample_rate, 2].flatten()

    # Compute automatic axis limits
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()
    z_min, z_max = Z.min(), Z.max()

    # Apply zoom-out factor
    x_range = (x_max - x_min) * zoom_out_factor
    y_range = (y_max - y_min) * zoom_out_factor
    z_range = (z_max - z_min) * zoom_out_factor

    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    z_center = (z_max + z_min) / 2

    ax.set_xlim([x_center - x_range / 2, x_center + x_range / 2])
    ax.set_ylim([y_center - y_range / 2, y_center + y_range / 2])
    ax.set_zlim([z_center - z_range / 2, z_center + z_range / 2])

    # Plot vectors
    ax.quiver(X, Y, Z, U, V, W, length=0.1, normalize=False, color='blue')

    # Plot particles
    ax.scatter(X, Y, Z, color='red', s=10, label="Particles")

    # Adjust camera view
    ax.view_init(elev=20, azim=45)
    ax.dist = 12  # Can be adjusted further if needed

    # Labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"{label}_Time_steps: " + str(time_steps) + ", nu = " + str(nu) + ", epochs = " + str(epochs_simulation) + ", realizations = " + str(realizations) + ", final time = " + str(final_time))
    ax.legend()
    plt.savefig(f"experiment_{label}.png")  # Save the plot as an image
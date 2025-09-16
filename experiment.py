import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(project_root)
import jax.numpy as jnp
import jax.random as random

from src.train import *

if __name__ == "__main__":

    grid_size = 30
    N = grid_size**3
    nu = 0.01
    N_time_steps = 3
    T = 0.5
    U, L = 1, 1
    N_realizations = 3
    epochs_simulation = 10000
    t = jnp.array([T])
    hidden_dim = 512
    learning_rate = 0.001
    decay_rate = 0.90

    ## Plotting the true solution
    grid = meshgrid(N).reshape((grid_size, grid_size, grid_size, 3))
    true_velocity = velocity_exact_solution_on_grid(t, grid, U, L=1, nu=nu)
    vorticity_true = -compute_minus_curl(true_velocity)
    plot_vector_field_projection((1/50)*vorticity_true, label="True")


    params = {
        "N": N,
        "N_steps": N_time_steps,
        "N_realizations": N_realizations,
        "T": T,
        "nu": nu,
        "hidden_dim": hidden_dim,
        "learning_rate": learning_rate,
        "num_epochs": epochs_simulation,
        "key": random.PRNGKey(42),
        "decay_rate": decay_rate,
    }

    positions, vorticity, loss_history = train(**params)

    positions = positions.reshape((N_realizations, grid_size, grid_size, grid_size, 3))
    vorticity = vorticity.reshape((N_realizations, grid_size, grid_size, grid_size, 3))

    plot_vector_field_projection((1/50)*vorticity.mean(axis=0), label="Ours")
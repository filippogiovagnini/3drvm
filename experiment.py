import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(project_root)
import jax.numpy as jnp
import jax.random as random

from src.train import *

if __name__ == "__main__":

    grid_size = 20
    N = grid_size**3
    nu = 0.01
    N_time_steps = 2
    T = 0.5
    U, L = 1, 1
    N_realizations = 5
    epochs_simulation = 15000
    t = jnp.array([T])
    # hidden_dim = 512 and learning_rate = 0.001 work well, do not change them.
    hidden_dim = 512
    learning_rate = 0.001
    decay_rate = 0.90 #0.65

    ## Plotting the true solution
    grid = meshgrid(N).reshape((grid_size, grid_size, grid_size, 3))
    true_velocity = velocity_exact_solution_on_grid(t, grid, U, L=1, nu=nu)
    plot_vector_field_projection((1/10)*true_velocity, label="True")

    ## Training the model
    positions, vorticity, loss_history  = train(N, N_time_steps, N_realizations, T, nu, hidden_dim, learning_rate, num_epochs = epochs_simulation, key = random.PRNGKey(42), decay_rate=decay_rate)
    positions = positions.reshape((N_realizations, grid_size, grid_size, grid_size, 3))
    vorticity = vorticity.reshape((N_realizations, grid_size, grid_size, grid_size, 3))

    ## From the approximated vorticity field back to the velocity field.
    k = 2 * jnp.pi / L
    velocity_field = (1/(jnp.sqrt(3) * k)) * vorticity

    ## Plot of the true velocity field.
    #plot_vectors(grid, true_velocity, sample_rate=2, zoom_out_factor=1.2, label="true_velocity")
    #plot_vectors(positions[0, ...], velocity_field[0, ...], sample_rate=2, zoom_out_factor=1.2, label="predicted_velocity") 
    plot_vector_field_projection((1/10)*velocity_field[0, ...], label="Ours")
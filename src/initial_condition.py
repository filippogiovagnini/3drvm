import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(project_root)

import jax
import jax.numpy as jnp
from num_solvers import *
from utilities import *

@jax.jit
def velocity_exact_solution(t: jnp.ndarray, X: jnp.ndarray, U: float, L: float, nu: float) -> jnp.ndarray:
    '''
    Exact solution for the periodic boundary condition problem. The domain is [-L, L]^3.
    Args:
        t: time variable. Shape (1,).
        X: spatial variable. Shape (3,).
    '''

    k = 2 * jnp.pi / (2 * L)

    #x = 2*(X[0] + L/2)
    #y = 2*(X[1] + L/2)
    #z = 2*(X[2] + L/2)

    x = X[0]
    y = X[1]
    z = X[2]

    u_x = U * (
        jnp.sin(k*x - jnp.pi/3) * jnp.cos(k*y + jnp.pi/3) * jnp.sin(k*z + jnp.pi/2) - jnp.cos(k*z - jnp.pi/3)*jnp.sin(k*x + jnp.pi/3)*jnp.sin(k*y + jnp.pi/2)
    ) * jnp.exp(-3*nu*(k**2)*t)

    u_y = U * (
        jnp.sin(k*y - jnp.pi/3) * jnp.cos(k*z + jnp.pi/3) * jnp.sin(k*x + jnp.pi/2) - jnp.cos(k*x - jnp.pi/3)*jnp.sin(k*y + jnp.pi/3)*jnp.sin(k*z + jnp.pi/2)
    ) * jnp.exp(-3*nu*(k**2)*t)

    u_z = U * (
        jnp.sin(k*z - jnp.pi/3) * jnp.cos(k*x + jnp.pi/3) * jnp.sin(k*y + jnp.pi/2) - jnp.cos(k*y - jnp.pi/3)*jnp.sin(k*z + jnp.pi/3)*jnp.sin(k*x + jnp.pi/2)
    ) * jnp.exp(-3*nu*(k**2)*t)

    return jnp.array([u_x, u_y, u_z])

@jax.jit
def velocity_exact_solution_on_grid(t: jnp.ndarray, pos_grid: jnp.ndarray, U: float, L: float, nu: float) -> jnp.ndarray:
    """
    Vectorized version of velocity_exact_solution.
    
    Args:
        t: time (shape (1,))
        pos_grid: array of shape (Nx, Ny, Nz, 3)
    
    Returns:
        vel_grid: array of shape (Nx, Ny, Nz, 3)
    """

    def eval_point(xyz):
        return velocity_exact_solution(t, xyz, U, L, nu)

    flat_pos = pos_grid.reshape(-1, 3)
    flat_vel = jax.vmap(eval_point)(flat_pos)
    return flat_vel.reshape(pos_grid.shape)

# just testing
if __name__ == "__main__":

    # Initialization
    U, L = 1, 1
    N_steps = 1
    T = 1
    N = 20**3
    dt = T / N_steps
    grid_size = round(N**(1/3))
    lattice = meshgrid(N)
    t = jnp.array([0.0])
    k = 2 * jnp.pi / L
    nu = 0.1

    initial_velocity_field = velocity_exact_solution_on_grid(t, lattice.reshape(grid_size, grid_size, grid_size, 3), U, L, nu)

    plot_vectors(lattice.reshape(grid_size, grid_size, grid_size, 3), initial_velocity_field)
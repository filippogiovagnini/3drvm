import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(project_root)

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg
import matplotlib.pyplot as plt
from interpolators import *

@jax.jit
def compute_minus_curl(F: jnp.ndarray) -> jnp.ndarray:
    """
    Compute minus the curl of a 3D vector field stored as an (grid_size, grid_size, grid_size, 3) JAX array.
    I the domain is [-1, 1]³, then if grid_size = 100 we have h = 2/100 = 0.02.

    Args:
        F: JAX array of shape (grid_size, grid_size, grid_size, 3), where F[..., 0] = Fx, F[..., 1] = Fy, F[..., 2] = Fz and where the positions of the grid are assumed to be (i, j, k) = (0, 0, 0), (1, 0, 0), ..., (grid_x - 1, grid_y - 1, grid_z - 1).

    Returns:
        C: JAX array of shape (grid_size, grid_size, grid_size, 3) representing the curl vector field.
    """

    grid_size = F.shape[0]
    h = 2/grid_size

    Fx, Fy, Fz = F[..., 0], F[..., 1], F[..., 2]
    dFz_dy = (jnp.roll(Fz, -1, axis=1) - jnp.roll(Fz, 1, axis=1)) / (2 * h)
    dFy_dz = (jnp.roll(Fy, -1, axis=2) - jnp.roll(Fy, 1, axis=2)) / (2 * h)

    dFx_dz = (jnp.roll(Fx, -1, axis=2) - jnp.roll(Fx, 1, axis=2)) / (2 * h)
    dFz_dx = (jnp.roll(Fz, -1, axis=0) - jnp.roll(Fz, 1, axis=0)) / (2 * h)

    dFy_dx = (jnp.roll(Fy, -1, axis=0) - jnp.roll(Fy, 1, axis=0)) / (2 * h)
    dFx_dy = (jnp.roll(Fx, -1, axis=1) - jnp.roll(Fx, 1, axis=1)) / (2 * h)

    # Compute -curl(vorticity) components
    Cx = - dFz_dy + dFy_dz
    Cy = - dFx_dz + dFz_dx
    Cz = - dFy_dx + dFx_dy

    return jnp.stack([Cx, Cy, Cz], axis=-1)

@jax.jit
def apply_laplacian(U_flat: jnp.ndarray) -> jnp.ndarray:
    """
    Applies the 3D Laplacian to a flattened array (vector field).

    Args:
        U_flat: Flattened 3D vector field of shape (N³, 3).
        grid_size: Size of the grid.

    Returns:
        L: Flattened Laplacian of U.
    """

    grid_size_cubed = U_flat.shape[0]  # Assuming U_flat is a flattened 3D vector field
    grid_size = int(round(grid_size_cubed ** (1/3))) 
    h = 2/grid_size
    U = U_flat.reshape((grid_size, grid_size, grid_size, 3))  # Reshape into 3D vector field
    L = -6 * U
    L += jnp.roll(U, 1, axis=0) + jnp.roll(U, -1, axis=0)
    L += jnp.roll(U, 1, axis=1) + jnp.roll(U, -1, axis=1)
    L += jnp.roll(U, 1, axis=2) + jnp.roll(U, -1, axis=2)
    return (L / h**2).reshape((-1, 3))  # Flatten back to 2D (grid_width³, 3)

@jax.jit
def velocity_from_vorticity(initial_vorticity: jnp.ndarray, h: float, grid_size: int) -> jnp.ndarray:
    """
    Compute the velocity field from the vorticity field using the Poisson equation.

    Args:
        initial_vorticity: Initial vorticity field. Shape (grid_size, grid_size, grid_size, 3). This must be the vorticity field computed in a h uniformly spaced grid with in the domain [-1, 1]³. This has to result in a shape (grid_size, grid_size, grid_size, 3).
        h: Grid spacing.
        grid_size: Grid size.

    Returns:
        U: Velocity field. Shape (grid_size, grid_size, grid_size, 3).
    """
    grid_size = initial_vorticity.shape[0]  # Assuming cubic grid
    F = compute_minus_curl(initial_vorticity)
    F_flat = F.reshape((-1, 3))  # Shape: (grid_size³, 3)

    # Solve using Conjugate Gradient (CG) with matrix-free multiplication
    u_flat, _ = cg(apply_laplacian, F_flat)

    # Reshape the solution back to (grid_size, grid_size, grid_size, 3)
    U = u_flat.reshape((grid_size, grid_size, grid_size, 3))
    return U

@jax.jit
def compute_nabla_u_prev(U: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the gradient of a vector field U on a 3D grid.

    Args:
        U (jnp.ndarray): Vector field of shape (grid_width, grid_width, grid_width, 3).
        h (float): Grid spacing.

    Returns:
        nabla_u_prev: Gradient tensor of shape (grid_width, grid_width, grid_width, 3, 3),
                     where the last two dimensions represent the gradient matrix:
                     grad_U[i, j, k, :, :] = [[dUx/dx, dUx/dy, dUx/dz],
                                              [dUy/dx, dUy/dy, dUy/dz],
                                              [dUz/dx, dUz/dy, dUz/dz]].
    """

    Ux, Uy, Uz = U[..., 0], U[..., 1], U[..., 2]

    grid_size = U.shape[0]
    h = 2 / grid_size

    dUx_dx = (jnp.roll(Ux, -1, axis=0) - jnp.roll(Ux, 1, axis=0)) / (2 * h)
    dUx_dy = (jnp.roll(Ux, -1, axis=1) - jnp.roll(Ux, 1, axis=1)) / (2 * h)
    dUx_dz = (jnp.roll(Ux, -1, axis=2) - jnp.roll(Ux, 1, axis=2)) / (2 * h)

    dUy_dx = (jnp.roll(Uy, -1, axis=0) - jnp.roll(Uy, 1, axis=0)) / (2 * h)
    dUy_dy = (jnp.roll(Uy, -1, axis=1) - jnp.roll(Uy, 1, axis=1)) / (2 * h)
    dUy_dz = (jnp.roll(Uy, -1, axis=2) - jnp.roll(Uy, 1, axis=2)) / (2 * h)

    dUz_dx = (jnp.roll(Uz, -1, axis=0) - jnp.roll(Uz, 1, axis=0)) / (2 * h)
    dUz_dy = (jnp.roll(Uz, -1, axis=1) - jnp.roll(Uz, 1, axis=1)) / (2 * h)
    dUz_dz = (jnp.roll(Uz, -1, axis=2) - jnp.roll(Uz, 1, axis=2)) / (2 * h)

    nabla_u_prev = jnp.stack([
        jnp.stack([dUx_dx, dUx_dy, dUx_dz], axis=-1),
        jnp.stack([dUy_dx, dUy_dy, dUy_dz], axis=-1),
        jnp.stack([dUz_dx, dUz_dy, dUz_dz], axis=-1)
    ], axis=-2)

    return nabla_u_prev

@jax.jit
def velocity_from_vorticity(initial_vorticity: jnp.ndarray, h: float, grid_size: int) -> jnp.ndarray:
    """
    Compute the velocity field from the vorticity field using the Poisson equation.
    Args:
        initial_vorticity: Initial vorticity field. Shape (grid_size, grid_size, grid_size, 3). This must be the vorticity field computed in a h uniformly spaced grid with in the domain [-1, 1]³. This has to result in a shape (grid_size, grid_size, grid_size, 3).
        h: Grid spacing.
    """
    grid_size = initial_vorticity.shape[0]
    F = compute_minus_curl(initial_vorticity)
    F_flat = F.reshape((-1, 3))  # Shape: (grid_size³, 3)

    # Solve using Conjugate Gradient (CG) with matrix-free multiplication
    u_flat, _ = cg(apply_laplacian, F_flat)

    # Reshape the solution back to (grid_size, grid_size, grid_size, 3)
    U = u_flat.reshape((grid_size, grid_size, grid_size, 3))
    return U

@jax.jit
def main_vel_grad_vel(vorticity_on_a_grid: jnp.ndarray, X: jnp.ndarray, grid_size: int = 1000000) -> tuple[jnp.ndarray, jnp.ndarray]:
    """"
    Given a vorticity field defined on a grid and particle positions, compute the velocity and its gradient at those positions.

    Args:
        vorticity: function. Takes a 3D position (x, y, z) and returns a 3D vector.
        X (N, 3): Particle positions.
        grid_size: Number of points along each axis.

    Returns:
        vorticity (N, 3): Vorticity field at particle positions.
        nabla_u (N, 3, 3): Gradient of the velocity field at particle positions.
    """

    h = 2 / grid_size

    U = velocity_from_vorticity(vorticity_on_a_grid, h, grid_size)

    nabla_u_prev = compute_nabla_u_prev(U)

    U_at_X = trilinear_interpolation_U(X, U)

    nabla_u_at_X = trilinear_interpolation_nabla_u_prev(X, nabla_u_prev)

    return U_at_X, nabla_u_at_X
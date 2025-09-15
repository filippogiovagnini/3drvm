import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(project_root)

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg
import matplotlib.pyplot as plt


@jax.jit
def trilinear_interpolation_U(X: jnp.ndarray, U: jnp.ndarray) -> jnp.ndarray:
    """
    Perform trilinear interpolation to compute velocities at particle positions.

    Args:
        X (jnp.ndarray): Particle positions of shape (m, N, 3).
        U (jnp.ndarray): Velocity field of shape (grid_width, grid_width, grid_width, 3).

    Returns:
        jnp.ndarray: Interpolated velocities at particle positions, shape (m, N, 3).
    """
    grid_size = U.shape[0]
    h = 2 / grid_size

    indices = (X + 1) / h     # We will have to modify this part if instead of [-1, 1]³ we use a different domain

    i0 = jnp.floor(indices).astype(int)
    i1 = i0 + 1

    w0 = i1 - indices  # Weight for lower corner
    w1 = indices - i0  # Weight for upper corner

    i0 = i0 % grid_size
    i1 = i1 % grid_size

    interpolated = (
        w0[..., 0:1] * w0[..., 1:2] * w0[..., 2:3] * U[i0[..., 0], i0[..., 1], i0[..., 2]] +
        w0[..., 0:1] * w0[..., 1:2] * w1[..., 2:3] * U[i0[..., 0], i0[..., 1], i1[..., 2]] +
        w0[..., 0:1] * w1[..., 1:2] * w0[..., 2:3] * U[i0[..., 0], i1[..., 1], i0[..., 2]] +
        w0[..., 0:1] * w1[..., 1:2] * w1[..., 2:3] * U[i0[..., 0], i1[..., 1], i1[..., 2]] +
        w1[..., 0:1] * w0[..., 1:2] * w0[..., 2:3] * U[i1[..., 0], i0[..., 1], i0[..., 2]] +
        w1[..., 0:1] * w0[..., 1:2] * w1[..., 2:3] * U[i1[..., 0], i0[..., 1], i1[..., 2]] +
        w1[..., 0:1] * w1[..., 1:2] * w0[..., 2:3] * U[i1[..., 0], i1[..., 1], i0[..., 2]] +
        w1[..., 0:1] * w1[..., 1:2] * w1[..., 2:3] * U[i1[..., 0], i1[..., 1], i1[..., 2]]
    )

    return interpolated

@jax.jit
def trilinear_interpolation_nabla_u_one_realization_of_X(X: jnp.ndarray, nabla_u_prev: jnp.ndarray) -> jnp.ndarray:
    """
    Perform trilinear interpolation to compute gradient tensors at particle positions.

    Args:
        X (jnp.ndarray): must be of shape (N, 3).
        nabla_u_prev (jnp.ndarray): Gradient tensor field of shape (grid_width, grid_width, grid_width, 3, 3).
        h (float): Grid spacing.

    Returns:
        jnp.ndarray: Interpolated gradient tensors at particle positions, shape (N, 3, 3).
    """
    grid_size = nabla_u_prev.shape[0]
    h = 2 / grid_size

    # Normalize particle positions to grid indices
    indices = (X + 1) / h

    i0 = jnp.floor(indices).astype(int)  # Lower corner indices
    i1 = i0 + 1  # Upper corner indices

    # Clip indices to stay within grid bounds
    #i0 = jnp.clip(i0, 0, nabla_u_prev.shape[0] - 1)
    #i1 = jnp.clip(i1, 0, nabla_u_prev.shape[0] - 1)

    # Compute weights for interpolation
    w0 = i1 - indices  # Weight for lower corner
    w1 = indices - i0  # Weight for upper corner

    i0 = i0 % grid_size
    i1 = i1 % grid_size

    # Perform trilinear interpolation
    interpolated = (
        w0[:, 0:1, None] * w0[:, 1:2, None] * w0[:, 2:3, None] * nabla_u_prev[i0[:, 0], i0[:, 1], i0[:, 2]] +
        w0[:, 0:1, None] * w0[:, 1:2, None] * w1[:, 2:3, None] * nabla_u_prev[i0[:, 0], i0[:, 1], i1[:, 2]] +
        w0[:, 0:1, None] * w1[:, 1:2, None] * w0[:, 2:3, None] * nabla_u_prev[i0[:, 0], i1[:, 1], i0[:, 2]] +
        w0[:, 0:1, None] * w1[:, 1:2, None] * w1[:, 2:3, None] * nabla_u_prev[i0[:, 0], i1[:, 1], i1[:, 2]] +
        w1[:, 0:1, None] * w0[:, 1:2, None] * w0[:, 2:3, None] * nabla_u_prev[i1[:, 0], i0[:, 1], i0[:, 2]] +
        w1[:, 0:1, None] * w0[:, 1:2, None] * w1[:, 2:3, None] * nabla_u_prev[i1[:, 0], i0[:, 1], i1[:, 2]] +
        w1[:, 0:1, None] * w1[:, 1:2, None] * w0[:, 2:3, None] * nabla_u_prev[i1[:, 0], i1[:, 1], i0[:, 2]] +
        w1[:, 0:1, None] * w1[:, 1:2, None] * w1[:, 2:3, None] * nabla_u_prev[i1[:, 0], i1[:, 1], i1[:, 2]]
    )

    return interpolated

@jax.jit
def trilinear_interpolation_nabla_u_prev(X: jnp.ndarray, nabla_u_prev: jnp.ndarray) -> jnp.ndarray:
    '''
    Perform trilinear interpolation to compute gradient tensors at particle positions for multiple realizations.

    Args:
        X (jnp.ndarray): Particle positions of shape for all the realizations, so must have shape (m, N, 3).
        nabla_u_prev (jnp.ndarray): Gradient tensor field of shape (grid_width, grid_width, grid_width, 3, 3).

    Returns:
        jnp.ndarray: Interpolated gradient tensors at particle positions, shape (m, N, 3, 3).
    '''
    
    if X.ndim == 3:
        interpolated = []
        for i in range(X.shape[0]):
            interpolated.append(trilinear_interpolation_nabla_u_one_realization_of_X(X[i], nabla_u_prev))
        return jnp.stack(interpolated, axis=0)  # Shape (m, N, 3, 3)
    else:
        return trilinear_interpolation_nabla_u_one_realization_of_X(X, nabla_u_prev)
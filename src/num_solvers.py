import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(project_root)

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg
import matplotlib.pyplot as plt

grid_size = 50000
h = 2/grid_size

def compute_minus_curl(F):
    """
    Compute minus the curl of a 3D vector field stored as an (grid_width, grid_width, grid_width, 3) JAX array.
    I the domain is [-1, 1]³, then if grid_width = 100 we have h = 2/100 = 0.02. 
    
    Args:
        F: JAX array of shape (grid_width, grid_width, grid_width, 3), where F[..., 0] = Fx, F[..., 1] = Fy, F[..., 2] = Fz and where the positions of the grid are assumed to be (i, j, k) = (0, 0, 0), (1, 0, 0), ..., (grid_x - 1, grid_y - 1, grid_z - 1).
        h: Grid spacing (assumed uniform in all directions).

    Returns:
        C: JAX array of shape (grid_width, grid_width, grid_width, 3) representing the curl vector field.
    """
    # Compute the grid spacing
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

def apply_laplacian(U_flat):
    """
    Applies the 3D Laplacian to a flattened array (vector field).
    Args:
        U_flat: Flattened 3D vector field of shape (N³, 3).
        grid_size: Size of the grid.
    Returns:
        L: Flattened Laplacian of U.
    """
    grid_size = U_flat.shape[0]  # Assuming U_flat is a flattened 3D vector field
    grid_size_2 = int(round(grid_size ** (1/3))) 
    U = U_flat.reshape((grid_size_2, grid_size_2, grid_size_2, 3))  # Reshape into 3D vector field
    L = -6 * U
    L += jnp.roll(U, 1, axis=0) + jnp.roll(U, -1, axis=0)
    L += jnp.roll(U, 1, axis=1) + jnp.roll(U, -1, axis=1)
    L += jnp.roll(U, 1, axis=2) + jnp.roll(U, -1, axis=2)
    return (L / h**2).reshape((-1, 3))  # Flatten back to 2D (grid_width³, 3)


def velocity_from_vorticity(initial_vorticity, h, grid_size):
    """
    Compute the velocity field from the vorticity field using the Poisson equation.
    Args:
        initial_vorticity: Initial vorticity field. Shape (grid_width, grid_width, grid_width, 3). This must be the vorticity field computed in a h uniformly spaced grid with in the domain [-1, 1]³. This has to result in a shape (grid_width, grid_width, grid_width, 3).
        h: Grid spacing.
    """
    grid_size = initial_vorticity.shape[0]  # Assuming cubic grid
    F = compute_minus_curl(initial_vorticity)
    F_flat = F.reshape((-1, 3))  # Shape: (grid_width³, 3)

    # Solve using Conjugate Gradient (CG) with matrix-free multiplication
    u_flat, _ = cg(apply_laplacian, F_flat)

    # Reshape the solution back to (grid_width, grid_width, grid_width, 3)
    U = u_flat.reshape((grid_size, grid_size, grid_size, 3))
    return U
    

def compute_nabla_u_prev(U):
    """
    Compute the gradient of a vector field U on a 3D grid.

    Args:
        U (jnp.ndarray): Vector field of shape (grid_width, grid_width, grid_width, 3).
        h (float): Grid spacing.

    Returns:
        jnp.ndarray: Gradient tensor of shape (grid_width, grid_width, grid_width, 3, 3),
                     where the last two dimensions represent the gradient matrix:
                     grad_U[i, j, k, :, :] = [[dUx/dx, dUx/dy, dUx/dz],
                                              [dUy/dx, dUy/dy, dUy/dz],
                                              [dUz/dx, dUz/dy, dUz/dz]].
    """
    # Extract components of the vector field
    Ux, Uy, Uz = U[..., 0], U[..., 1], U[..., 2]

    grid_size = U.shape[0]  # Assuming cubic grid
    h = 2 / grid_size  # Grid spacing

    # Compute partial derivatives using central differences
    dUx_dx = (jnp.roll(Ux, -1, axis=0) - jnp.roll(Ux, 1, axis=0)) / (2 * h)
    dUx_dy = (jnp.roll(Ux, -1, axis=1) - jnp.roll(Ux, 1, axis=1)) / (2 * h)
    dUx_dz = (jnp.roll(Ux, -1, axis=2) - jnp.roll(Ux, 1, axis=2)) / (2 * h)

    dUy_dx = (jnp.roll(Uy, -1, axis=0) - jnp.roll(Uy, 1, axis=0)) / (2 * h)
    dUy_dy = (jnp.roll(Uy, -1, axis=1) - jnp.roll(Uy, 1, axis=1)) / (2 * h)
    dUy_dz = (jnp.roll(Uy, -1, axis=2) - jnp.roll(Uy, 1, axis=2)) / (2 * h)

    dUz_dx = (jnp.roll(Uz, -1, axis=0) - jnp.roll(Uz, 1, axis=0)) / (2 * h)
    dUz_dy = (jnp.roll(Uz, -1, axis=1) - jnp.roll(Uz, 1, axis=1)) / (2 * h)
    dUz_dz = (jnp.roll(Uz, -1, axis=2) - jnp.roll(Uz, 1, axis=2)) / (2 * h)

    # Combine partial derivatives into the gradient tensor
    nabla_u_prev = jnp.stack([
        jnp.stack([dUx_dx, dUx_dy, dUx_dz], axis=-1),
        jnp.stack([dUy_dx, dUy_dy, dUy_dz], axis=-1),
        jnp.stack([dUz_dx, dUz_dy, dUz_dz], axis=-1)
    ], axis=-2)

    return nabla_u_prev

def main_vel_grad_vel(vorticity_on_a_grid, X, grid_size=1000000):
    """"
    Args:
        vorticity: function. Takes a 3D position (x, y, z) and returns a 3D vector.
        X (N, 3): Particle positions.
        grid_width: Number of points along each axis.
    Output:
        vorticity (N, 3): Vorticity field at particle positions.
        nabla_u (N, 3, 3): Gradient of the velocity field at particle positions.
    """

    # Grid spacing
    h = 2 / grid_size

    # Compute the velocity field
    U = velocity_from_vorticity(vorticity_on_a_grid, h, grid_size)

    # Compute the gradient of the velocity field
    nabla_u_prev = compute_nabla_u_prev(U)

    U_at_X = trilinear_interpolation_U(X, U)

    # Compute the gradient of the velocity field at particle positions
    nabla_u_at_X = trilinear_interpolation_nabla_u_prev(X, nabla_u_prev)

    return U_at_X, nabla_u_at_X


def trilinear_interpolation_U(X, U):
    """
    Perform trilinear interpolation to compute velocities at particle positions.

    Args:
        X (jnp.ndarray): Particle positions of shape (m, N, 3).
        U (jnp.ndarray): Velocity field of shape (grid_width, grid_width, grid_width, 3).

    Returns:
        jnp.ndarray: Interpolated velocities at particle positions, shape (m, N, 3).
    """
    grid_size = U.shape[0]  # Assuming cubic grid
    h = 2 / grid_size  # Grid spacing

    # We will have to modify this part if instead of [-1, 1]³ we use a different domain
    indices = (X + 1) / h

    i0 = jnp.floor(indices).astype(int)  # Lower corner indices
    i1 = i0 + 1  # Upper corner indices

    # Clip indices to stay within grid bounds
    #i0 = jnp.clip(i0, 0, U.shape[0] - 1)
    #i1 = jnp.clip(i1, 0, U.shape[0] - 1)

    # Compute weights for interpolation
    w0 = i1 - indices  # Weight for lower corner
    w1 = indices - i0  # Weight for upper corner

    i0 = i0 % grid_size
    i1 = i1 % grid_size

    # Perform trilinear interpolation
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

def trilinear_interpolation_nabla_u_one_realization_of_X(X, nabla_u_prev):
    """
    Perform trilinear interpolation to compute gradient tensors at particle positions.

    Args:
        X (jnp.ndarray): must be of shape (N, 3).
        nabla_u_prev (jnp.ndarray): Gradient tensor field of shape (grid_width, grid_width, grid_width, 3, 3).
        h (float): Grid spacing.

    Returns:
        jnp.ndarray: Interpolated gradient tensors at particle positions, shape (N, 3, 3).
    """
    grid_size = nabla_u_prev.shape[0]  # Assuming cubic grid
    h = 2 / grid_size  # Grid spacing
    
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

def trilinear_interpolation_nabla_u_prev(X, nabla_u_prev):
    '''
    Args:
        X (jnp.ndarray): Particle positions of shape for all the realizations, so must have shape (m, N, 3).
    '''
    if X.ndim == 3:
        interpolated = []
        for i in range(X.shape[0]):
            interpolated.append(trilinear_interpolation_nabla_u_one_realization_of_X(X[i], nabla_u_prev))
        return jnp.stack(interpolated, axis=0)  # Shape (m, N, 3, 3)
    else:
        return trilinear_interpolation_nabla_u_one_realization_of_X(X, nabla_u_prev)


def meshgrid(N, xmin=-1, xmax=1, ymin=-1, ymax=1, zmin=-1, zmax=1):
    """
    Generate a 3D mesh grid of particle positions.
    N: Number of particles. (it must be a cube of an integer)

    Returns:
        X (N, 3): Particle positions in 3D space, where N is the total number of points.
    """
    grid_size = round(N**(1/3)) # Number of points along each axis
    x = jnp.linspace(xmin, xmax, grid_size)
    y = jnp.linspace(ymin, ymax, grid_size)
    z = jnp.linspace(zmin, zmax, grid_size)

    # Create a 3D mesh grid
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')

    # Flatten the grid into a list of 3D points
    positions = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    return positions


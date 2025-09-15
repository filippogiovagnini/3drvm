import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(project_root)

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

@partial(jit, static_argnums=[5])
def update_step_X(X: jnp.ndarray, U: jnp.ndarray, dt: float, nu: float, key: jax.random.PRNGKey, m: int) -> jnp.ndarray:
    """
    Update particle positions using an Euler–Maruyama step.

    Args:
        X (m, N, d): Current particle positions.
        U (m, N, d): Drift velocity for each particle.
        dt (scalar): Time step.
        nu (scalar): Diffusion coefficient.
        key: JAX PRNGKey for random number generation.
        m: number of Monte Carlo estimates

    Returns:
        X_next (m, N, d): Updated particle positions.
    """
    m, N, d = X.shape[0], X.shape[1],  X.shape[2]
    noise = jax.random.normal(key=key, shape=(m, N, d))
    X_next = X + dt * U + jnp.sqrt(2 * nu * dt) * noise
    X_next = 2 * ((X_next + 1)/2 % 1) - 1        # Periodic boundary conditions
    return X_next

@jax.jit
def update_step_G(nabla_u_prev: jnp.ndarray, dt: float) -> jnp.ndarray:
    """
    Solve the G ODE via backward Euler integration, then reverse the time axis to recover forward ordering.

    Args:
        nabla_u_prev: (m, T, N, d, d) — time series of velocity gradient matrices.
        dt: Scalar time step.

    Returns:
        G_ts (T, m, N, d, d): Time evolution of G (ordered forward in time).
    """
    def backward_step(current_G: jnp.ndarray, inputs):
        """
        Perform one backward Euler update step for G.

        Args:
            current_G (m, N, d, d): The current G value.
            inputs: Tuple containing:
                - nabla_u (m, N, d, d): Current time-step velocity gradient.

        Returns:
            new_G (m, N, d, d): Updated G value.
            new_G (m, N, d, d): Value to record for this time step.
        """
        nabla_u = inputs
        update = jnp.matmul(current_G, nabla_u)
        new_G = current_G - (-dt) * update
        return new_G, new_G

    T, m, N, d, _ = nabla_u_prev.shape

    G_initial = jnp.tile(jnp.eye(d), (m, N, 1, 1))
    final_G, G_series = jax.lax.scan(backward_step, G_initial, nabla_u_prev)
    return final_G

@jax.jit
def update_Omega(G_t0: jnp.ndarray, vorticity_0: jnp.ndarray) -> jnp.ndarray:
    """
    Update Omega based on the evolution of G, the function g(X), and the indicator D.

    Args:
        G_t0: (m, N, d, d) — G(t,0).
        vorticity_0: (N, d) — Initial vorticity.

    Returns:
        Omega (m, N, d): Updated Omega values.
    """
    Omega = jnp.matmul(G_t0, vorticity_0[:, :, :, None]).squeeze(-1)
    return Omega
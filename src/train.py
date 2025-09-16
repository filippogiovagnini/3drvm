import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(project_root)

from num_solvers import *
from interpolators import *
from update_stepper import *
from initial_condition import *
import optax
import jax
import jax.numpy as jnp
from jax import random



def train(N, N_steps, N_realizations, T, nu = 0.1, hidden_dim = 10, learning_rate = 0.01, num_epochs = 1000000, key = jax.random.PRNGKey(42), decay_rate = 0.95):
    """
    Train the neural network model to learn the vorticity field.

    Args:
        N (int): Number of particles. (it must be a cube of an integer)
        N_steps (int): Number of time steps.
        N_realizations (int): Number of realizations of the stochastic processes.
        nu (float): Viscosity coefficient.
        hidden_dim (int): Number of hidden units in the neural network.
        learning_rate (float): Learning rate for the optimizer.
        T (float): Total time.
        vorticity_0 (callable): Initial vorticity field.
    Returns:
        positions (jnp.ndarray): Array of shape (N, 3) containing 3D points.
    """

    # A helper function to randomly initialize weights and biases
    # for a dense neural network layer
    def random_layer_params(m, n, key, scale=1e-2):
        w_key, b_key = random.split(key)
        return scale * random.normal(w_key, (m, n)), scale * random.normal(b_key, (n,))

    # Initialize all layers for a fully-connected neural network with sizes "sizes"
    def init_params(key, input_dim, hidden_dim, output_dim=3):
        sizes = [input_dim, hidden_dim, hidden_dim, hidden_dim, output_dim]
        keys = random.split(key, len(sizes))
        return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]
    
    def VorticityNN(params, X):
        # per-example predictions
        activations = X
        for w, b in params[:-1]:
            outputs = jnp.dot(activations, w) + b
            activations = jax.nn.selu(outputs)

        final_w, final_b = params[-1]
        output = jnp.dot(activations, final_w) + final_b
        return output 

    ### JUST ADDED
    def periodic_penalty(params, batch_size=64):
        # Make a uniform grid in [-1,1] for boundary sampling
        coords = jnp.linspace(-1., 1., batch_size).reshape(-1,1)

        # Build mesh for y,z pairs
        y, z = jnp.meshgrid(coords.flatten(), coords.flatten(), indexing="ij")
        y = y.reshape(-1,1)
        z = z.reshape(-1,1)
        x = coords  # reuse for x-boundary cases

        # x-boundary: (±1, y, z)
        pos = jnp.hstack([ jnp.ones_like(y), y, z ])
        neg = jnp.hstack([-jnp.ones_like(y), y, z ])
        loss_x = jnp.mean((VorticityNN(params, pos) - VorticityNN(params, neg))**2)

        # y-boundary: (x, ±1, z)
        pos = jnp.hstack([ x, jnp.ones_like(x), z[:len(x)] ])  # match shapes
        neg = jnp.hstack([ x,-jnp.ones_like(x), z[:len(x)] ])
        loss_y = jnp.mean((VorticityNN(params, pos) - VorticityNN(params, neg))**2)

        # z-boundary: (x, y, ±1)
        pos = jnp.hstack([ x, y[:len(x)], jnp.ones_like(x) ])
        neg = jnp.hstack([ x, y[:len(x)],-jnp.ones_like(x) ])
        loss_z = jnp.mean((VorticityNN(params, pos) - VorticityNN(params, neg))**2)

        return loss_x + loss_y + loss_z



    '''    # Define loss function given by Eq. (1) in the document
    def loss_function(params, eta, X, Omega, Delta_eta):
        N = X.shape[1]
        w_eta = VorticityNN(params, eta)
        w_X = VorticityNN(params, X)
        #loss = jnp.multiply(w_eta**2, Delta_eta) - 1/N * 2 * jnp.multiply(Omega, w_X)
        loss = w_eta**2 -  jnp.mean(2 * jnp.multiply(Omega, w_X), axis = 0)
        #loss = (w_X-Omega)**2
        return jnp.mean(loss)
    '''
    

    def loss_function(params, eta, X, Omega, Delta_eta, lambda_bc=1.0):
        N = X.shape[1]
        w_eta = VorticityNN(params, eta)
        w_X   = VorticityNN(params, X)

        # Original term
        core_loss = jnp.mean(w_eta**2 - jnp.mean(2 * jnp.multiply(Omega, w_X), axis=0))

        # Periodic boundary penalty
        bc_loss = periodic_penalty(params, batch_size=64)  # pass in PRNG key properly

        return core_loss + lambda_bc * bc_loss
    
    @jax.jit
    def update(params, opt_state, eta, X, Omega, Delta_eta):
        loss, grads = jax.value_and_grad(loss_function)(params, eta, X, Omega, Delta_eta)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Initialization
    U, L = 1, 1
    dt = T / N_steps
    grid_size = round(N**(1/3))
    lattice = meshgrid(N)
    t = jnp.array([0.0])
    k = 2 * jnp.pi / L

    initial_velocity_field = velocity_exact_solution_on_grid(t, lattice.reshape(grid_size, grid_size, grid_size, 3), U, L, nu)
    vorticity_0_X = -compute_minus_curl(initial_velocity_field)
    vorticity_0_X = vorticity_0_X.reshape((grid_size**3, 3))  # Reshape to (N, 3)

    size1 = vorticity_0_X.shape[0]
    size2 = 3

    # This has shape (m, size1, 3)
    Omega_0 = vorticity_0_X

    nabla_u_history = []

    input_dim = 3
    output_dim = 3
    params = init_params(key, input_dim, hidden_dim, output_dim)

    schedule = optax.exponential_decay(
        init_value=learning_rate, 
        transition_steps=1000,
        decay_rate=decay_rate
    )

    optimizer = optax.adam(schedule)
    #opt_state = optimizer.init(params)

    loss_history = jnp.zeros((N_steps, int(num_epochs/100)))
    

    def step_training(i, val, params):
        X, vorticity, Omega, rng = val

        rng, key = jax.random.split(rng)
        loss_history_step = []
        

        U, L = 1, 1
        dt = T / N_steps
        Delta_eta = N**(-1/3)
        lattice = meshgrid(N)
        k = 2 * jnp.pi / L

        # STEP 1: define and optimise loss function to compute the vorticity field
        #optimizer = optax.adam(learning_rate)


        if i == 0:
            num_epochs_step = num_epochs
            schedule = optax.exponential_decay(
                init_value=learning_rate, 
                transition_steps=1000,
                decay_rate=0.92
            )
            optimizer = optax.adam(schedule)

        else:
            num_epochs_step = int(num_epochs / 1)
            schedule = optax.exponential_decay(
                init_value=learning_rate, 
                transition_steps=1000,
                decay_rate=0.92
            )
            optimizer = optax.adam(schedule)

        opt_state = optimizer.init(params)

        # Apply training loop
        for epoch in range(num_epochs_step):
            params, opt_state, loss = update(params, opt_state, lattice, X, Omega, Delta_eta)
            #loss_history[i, epoch] = loss

            lattice_for_error = meshgrid(1000000)

            if epoch % 100 == 0:

                t_ = jnp.array([i * dt])

                velocity_field_at_t = velocity_exact_solution_on_grid(t_, lattice_for_error.reshape(100, 100, 100, 3), U, L, nu)

                vorticity_at_t = -compute_minus_curl(velocity_field_at_t).reshape((1000000, 3))
                error_norm = jnp.mean((VorticityNN(params, lattice_for_error) - vorticity_at_t)**2)
                norm_of_vorticity = jnp.mean(vorticity_at_t**2)
                rel_error_norm = jnp.sqrt(error_norm) / jnp.sqrt(norm_of_vorticity)
                #error_norm = jnp.sum((VorticityNN(params, X) - vorticity_0_X)**2)
                loss_history_step.append(loss)
                print(f"Time step: {i * dt}, Epoch {epoch}, Relative error is {int(100*rel_error_norm)}, Error_norm: {error_norm}, LR is {schedule(epoch)},Loss: {loss}")
                

        vorticity = VorticityNN(params, X)

        ## STEP 2: Velocity field
        # I need first to compute the vorticity field on a grid
        grid_size = 120
        lattice_new = meshgrid(grid_size**3)
        vorticity_on_a_grid = VorticityNN(params, lattice_new)
        
        vorticity_on_a_grid = vorticity_on_a_grid.reshape((grid_size, grid_size, grid_size, 3))
        # Compute the velocity field and its gradient
        X_flat = X.reshape((-1, 3))  # Flatten the positions for the velocity computation
        U, nabla_U = main_vel_grad_vel(vorticity_on_a_grid, X)
        
        nabla_U = nabla_U.reshape((N_realizations, N, 3, 3))  # Reshape to match the number of realizations and particles
        U = U.reshape((N_realizations, N, 3))  # Reshape to match the number of realizations and particles
        
        nabla_u_history.append(nabla_U)
        
        ## STEP 3: Update the positions of the particles
        X = update_step_X(X, U, dt, nu, key, 10) 
        G_t0 = update_step_G(jnp.stack(nabla_u_history, axis=0), dt)

        Omega = update_Omega(G_t0, jnp.tile(vorticity_0_X, (N_realizations, 1, 1)))

        return X, vorticity, Omega, key, params, loss_history_step
    
    positions = jnp.tile(lattice, (N_realizations, 1, 1))  
    vorticity = jnp.tile(vorticity_0_X, (N_realizations, 1, 1))

    for i in range(N_steps):
        positions, vorticity, Omega_0, key, params, loss_history_step = step_training(i, (positions, vorticity, Omega_0, key), params)
        loss_history = loss_history.at[i, :].set(loss_history_step)

    return positions, vorticity, loss_history
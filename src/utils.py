import jax
import jax.numpy as jnp

import numpy as np
from flax import nnx

import matplotlib.pyplot as plt

# ------------ Function Fitting -----------------------------------------

def generate_func_data(function, dim, N, seed):
    key = jax.random.key(seed)
    x = jax.random.uniform(key, shape=(N,dim), minval=-1.0, maxval=1.0)

    y = function(x)

    return x, y


@nnx.jit
def func_fit_step(model, optimizer, X_train, y_train):

    def loss_fn(model):
        residual = model(X_train) - y_train
        loss = jnp.mean((residual)**2)

        return loss

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)

    return loss


def func_fit_eval(model, function, dim, resolution=200, make_plot=False):
    # Create grid
    lin = jnp.linspace(-1.0, 1.0, resolution)
    xx = jnp.meshgrid(*[lin]*dim)
    grid = jnp.stack([x.ravel() for x in xx], axis=-1)

    # Evaluate ground truth and prediction
    y_true = function(grid)
    y_pred = model(grid)

    # Compute relative L2 error
    error = jnp.linalg.norm(y_true - y_pred) / jnp.linalg.norm(y_true)

    # Optionally make plot (2D only)
    if make_plot and dim == 2:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        shape = (resolution, resolution)
        
        vmin = y_true.min()
        vmax = y_true.max()
        abs_err = jnp.abs(y_true - y_pred)

        im0 = axs[0].imshow(y_true.reshape(shape), origin='lower', extent=[-1, 1, -1, 1], vmin=vmin, vmax=vmax)
        im1 = axs[1].imshow(y_pred.reshape(shape), origin='lower', extent=[-1, 1, -1, 1], vmin=vmin, vmax=vmax)
        im2 = axs[2].imshow(abs_err.reshape(shape), origin='lower', extent=[-1, 1, -1, 1])

        axs[0].set_title("Reference")
        axs[1].set_title("Predicted")
        axs[2].set_title("Absolute Difference")

        for ax in axs:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_xticks(jnp.linspace(-1, 1, 5))
            ax.set_yticks(jnp.linspace(-1, 1, 5))

        fig.colorbar(im0, ax=axs[0])
        fig.colorbar(im1, ax=axs[1])
        fig.colorbar(im2, ax=axs[2])

        plt.tight_layout()
        plt.show()

    return error

# ------------ Forward PDE problems -------------------------------------

def get_collocs(pde_name, N):

    if pde_name == "allen-cahn":
        
        # PDE collocation points
        t_pde = jnp.linspace(0, 1, N)
        x_pde = jnp.linspace(-1, 1, N)
        T_pde, X_pde = jnp.meshgrid(t_pde, x_pde, indexing='ij')
        pde_collocs = jnp.stack([T_pde.flatten(), X_pde.flatten()], axis=1) # (N^2, 2)
    
        # Initial condition: x^2 cos(πx)
        t_ic = jnp.array([0.0])
        T_ic, X_ic = jnp.meshgrid(t_ic, x_pde, indexing='ij')
        ic_collocs = jnp.stack([T_ic.flatten(), X_ic.flatten()], axis=1) # (N, 2)
        ic_data = ((ic_collocs[:,1]**2)*jnp.cos(jnp.pi*ic_collocs[:,1])).reshape(-1,1) # (N, 1)

        # Boundary conditions: u(t,-1) = u(t,1) = -1
        x_bc_1 = jnp.array([-1.0])
        T_bc_1, X_bc_1 = jnp.meshgrid(t_pde, x_bc_1, indexing='ij')
        bc_1 = jnp.stack([T_bc_1.flatten(), X_bc_1.flatten()], axis=1) # (N, 2)
        bc_1_data = -jnp.ones(bc_1.shape[0]).reshape(-1,1) # (N, 1)

        x_bc_2 = jnp.array([1.0])
        T_bc_2, X_bc_2 = jnp.meshgrid(t_pde, x_bc_2, indexing='ij')
        bc_2 = jnp.stack([T_bc_2.flatten(), X_bc_2.flatten()], axis=1) # (N, 2)
        bc_2_data = -jnp.ones(bc_2.shape[0]).reshape(-1,1) # (N, 1)

        bc_collocs = jnp.concatenate([ic_collocs, bc_1, bc_2], axis=0)
        bc_data = jnp.concatenate([ic_data, bc_1_data, bc_2_data], axis=0)
        
    elif pde_name == "burgers":

        # PDE collocation points
        t_pde = jnp.linspace(0, 1, N)
        x_pde = jnp.linspace(-1, 1, N)
        T_pde, X_pde = jnp.meshgrid(t_pde, x_pde, indexing='ij')
        pde_collocs = jnp.stack([T_pde.flatten(), X_pde.flatten()], axis=1) # (N^2, 2)
    
        # Initial condition: - sin(πx)
        t_ic = jnp.array([0.0])
        T_ic, X_ic = jnp.meshgrid(t_ic, x_pde, indexing='ij')
        ic_collocs = jnp.stack([T_ic.flatten(), X_ic.flatten()], axis=1) # (N, 2)
        ic_data = -jnp.sin(jnp.pi*ic_collocs[:,1]).reshape(-1,1) # (N, 1)

        # Boundary conditions: u(t,-1) = u(t,1) = 0
        x_bc_1 = jnp.array([-1.0])
        T_bc_1, X_bc_1 = jnp.meshgrid(t_pde, x_bc_1, indexing='ij')
        bc_1 = jnp.stack([T_bc_1.flatten(), X_bc_1.flatten()], axis=1) # (N, 2)
        bc_1_data = jnp.zeros(bc_1.shape[0]).reshape(-1,1) # (N, 1)

        x_bc_2 = jnp.array([1.0])
        T_bc_2, X_bc_2 = jnp.meshgrid(t_pde, x_bc_2, indexing='ij')
        bc_2 = jnp.stack([T_bc_2.flatten(), X_bc_2.flatten()], axis=1) # (N, 2)
        bc_2_data = jnp.zeros(bc_2.shape[0]).reshape(-1,1) # (N, 1)

        bc_collocs = jnp.concatenate([ic_collocs, bc_1, bc_2], axis=0)
        bc_data = jnp.concatenate([ic_data, bc_1_data, bc_2_data], axis=0)

    elif pde_name == "helmholtz":
        # PDE collocation points
        x_pde = jnp.linspace(-1, 1, N)
        y_pde = jnp.linspace(-1, 1, N)
        X_pde, Y_pde = jnp.meshgrid(x_pde, y_pde, indexing='ij')
        pde_collocs = jnp.stack([X_pde.flatten(), Y_pde.flatten()], axis=1) # (N^2, 2)
    
        # Boundary conditions: u(-1,y) = u(1,y) = u(x,-1) = u(x,1) = 0
        x_bc_1 = jnp.array([-1.0])
        X_bc_1, Y_bc_1 = jnp.meshgrid(x_bc_1, y_pde, indexing='ij')
        bc_1 = jnp.stack([X_bc_1.flatten(), Y_bc_1.flatten()], axis=1) # (N, 2)
        bc_1_data = jnp.zeros(bc_1.shape[0]).reshape(-1,1) # (N, 1)

        x_bc_2 = jnp.array([1.0])
        X_bc_2, Y_bc_2 = jnp.meshgrid(x_bc_2, y_pde, indexing='ij')
        bc_2 = jnp.stack([X_bc_2.flatten(), Y_bc_2.flatten()], axis=1) # (N, 2)
        bc_2_data = jnp.zeros(bc_2.shape[0]).reshape(-1,1) # (N, 1)

        y_bc_3 = jnp.array([-1.0])
        X_bc_3, Y_bc_3 = jnp.meshgrid(x_pde, y_bc_3, indexing='ij')
        bc_3 = jnp.stack([X_bc_3.flatten(), Y_bc_3.flatten()], axis=1) # (N, 2)
        bc_3_data = jnp.zeros(bc_3.shape[0]).reshape(-1,1) # (N, 1)

        y_bc_4 = jnp.array([1.0])
        X_bc_4, Y_bc_4 = jnp.meshgrid(x_pde, y_bc_4, indexing='ij')
        bc_4 = jnp.stack([X_bc_4.flatten(), Y_bc_4.flatten()], axis=1) # (N, 2)
        bc_4_data = jnp.zeros(bc_4.shape[0]).reshape(-1,1) # (N, 1)

        bc_collocs = jnp.concatenate([bc_1, bc_2, bc_3, bc_4], axis=0)
        bc_data = jnp.concatenate([bc_1_data, bc_2_data, bc_3_data, bc_4_data], axis=0)

    return pde_collocs, bc_collocs, bc_data


def get_ref(pde_name):

    ref = np.load(f'data/{pde_name}.npz')
    refsol = jnp.array(ref['usol'])

    N_t, N_x = ref['usol'].shape

    if pde_name != "helmholtz":
    
        t = ref['t'].flatten()
        x = ref['x'].flatten()
        T, X = jnp.meshgrid(t, x, indexing='ij')
        coords = jnp.stack([T.flatten(), X.flatten()], axis=1)

    else:

        x = ref['x'].flatten()
        y = ref['y'].flatten()
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        coords = jnp.stack([X.flatten(), Y.flatten()], axis=1)

    return refsol, coords
    

# ------------ Feynman Benchmarks ---------------------------------------

def generate_feyn_data(function, dim, N, seed):
    key = jax.random.key(seed)

    eps = 1e-6
    x = jax.random.uniform(key, shape=(N,dim), minval=-1.0+eps, maxval=1.0)
    x = jnp.where(jnp.abs(x) < eps, jnp.sign(x) * eps, x)

    y = function(x)

    return x, y


def feyn_fit_eval(model, function, dim):
    if dim == 1:
        resolution = 1000
    elif dim == 2:
        resolution = 200
    elif dim == 3:
        resolution = 30
    else:
        resolution = 10
        
    # Create grid
    eps = 1e-6
    lin = jnp.linspace(-1.0+eps, 1.0, resolution)
    lin = jnp.where(jnp.abs(lin) < eps, jnp.sign(lin) * eps, lin)
    
    xx = jnp.meshgrid(*[lin]*dim)
    grid = jnp.stack([x.ravel() for x in xx], axis=-1)

    # Evaluate ground truth and prediction
    y_true = function(grid)
    y_pred = model(grid)

    # Compute relative L2 error
    error = jnp.linalg.norm(y_true - y_pred) / jnp.linalg.norm(y_true)

    return error
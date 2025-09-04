import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from flax import nnx


def _grad_flat_at_x(model, x_single):
    # scalarize output by summing over channels (gives NTK = sum_c ∂f_c/∂θ · ∂f_c/∂θ')
    def scalar_out(m):
        y = m(x_single[None, ...])           # (1, C)
        return jnp.sum(y)                    # scalar
    g = nnx.grad(scalar_out)(model)          # pytree of grads wrt model params
    g_flat, _ = ravel_pytree(g)
    return g_flat


def ntk_matrix(model, X_subset):
    J = jax.vmap(lambda x: _grad_flat_at_x(model, x))(X_subset)
    return J @ J.T


def stabilize_kernel(K, jitter_scale=1e-12):
    # Symmetrize + trace-scaled jitter to kill tiny negative eigs
    K = 0.5 * (K + K.T)
    eps = jitter_scale * (jnp.trace(K) / K.shape[0])
    return K + eps * jnp.eye(K.shape[0], dtype=K.dtype)


def ntk_spectrum(model, X_subset, jitter_scale=1e-12):
    K = ntk_matrix(model, X_subset)
    K = stabilize_kernel(K, jitter_scale)
    ev = jnp.linalg.eigvalsh(K)
    return jnp.sort(ev)[::-1]


def cond_from_eigs(eigs, rel_floor=1e-9):
    lam_max = eigs[0]
    lam_min = jnp.maximum(eigs[-1], rel_floor * lam_max)
    return float(lam_max / lam_min)

# ------ PDE STUFF -------------------------------------

def _grad_flat_pde_at_x_weighted(model, pde_res_fn, x_single, w):
    def scalar_res(m):
        r = pde_res_fn(m, x_single[None, ...])  # (1,1) or (1,)
        return w * jnp.squeeze(r)               # weight * residual
    g = nnx.grad(scalar_res)(model)
    g_flat, _ = ravel_pytree(g)
    return g_flat

def _grad_flat_bc_at_xy_weighted(model, x_single, y_single, w):
    def scalar_res(m):
        r = m(x_single[None, ...]) - y_single[None, ...]  # (1,1) or (1,)
        return w * jnp.squeeze(r)
    g = nnx.grad(scalar_res)(model)
    g_flat, _ = ravel_pytree(g)
    return g_flat

def ntk_pde_matrix_weighted(model, pde_res_fn, X_pde, w_E):
    w_E = jnp.ravel(w_E)
    rows = [_grad_flat_pde_at_x_weighted(model, pde_res_fn, X_pde[i], w_E[i])
            for i in range(X_pde.shape[0])]
    J = jnp.stack(rows, axis=0)
    return J @ J.T

def ntk_bc_matrix_weighted(model, X_bc, Y_bc, w_B):
    w_B = jnp.ravel(w_B)
    rows = [_grad_flat_bc_at_xy_weighted(model, X_bc[i], Y_bc[i], w_B[i])
            for i in range(X_bc.shape[0])]
    J = jnp.stack(rows, axis=0)
    return J @ J.T


def pinntk_diag_spectra_weighted(model, pde_res_fn, X_pde, X_bc, Y_bc, w_E, w_B, jitter_scale=1e-12):
    K_EE = stabilize_kernel(ntk_pde_matrix_weighted(model, pde_res_fn, X_pde, w_E), jitter_scale)
    K_BB = stabilize_kernel(ntk_bc_matrix_weighted(model, X_bc, Y_bc, w_B), jitter_scale)
    lamE = jnp.sort(jnp.linalg.eigvalsh(K_EE))[::-1]
    lamB = jnp.sort(jnp.linalg.eigvalsh(K_BB))[::-1]
    return lamE, lamB
    
import jax.numpy as jnp
from jax.scipy.special import i1, i1e, fresnel, erfinv, erf


# Define the functions used for the task

def f1(x):
    return x[:, [0]] * x[:, [1]]

def f2(x):
    return jnp.exp(jnp.sin(jnp.pi * x[:, [0]]) + x[:, [1]]**2)

def f3(x):
    return i1(x[:, [0]]) + jnp.exp(i1e(x[:, [1]])) + jnp.sin(x[:, [0]] * x[:, [1]])

def f4(x):
    S, C = fresnel(f3(x) + erfinv(x[:, [1]]))
    return S * C

def f5(x):
    return x[:, 1].reshape(-1, 1) * jnp.where(x[:, 0] < 0.5, 1, -1).reshape(-1, 1) + erf(x[:, 0]).reshape(-1, 1) * jnp.where(f1(x) < 1, f1(x), 1/f1(x))
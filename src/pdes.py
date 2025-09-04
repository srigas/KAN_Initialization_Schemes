import jax.numpy as jnp

from jaxkan.utils.PIKAN import gradf


# Allen-Cahn Equation
def ac_res(model, collocs):
    
    # Eq. parameters
    D = jnp.array(1e-4, dtype=jnp.float32)
    c = jnp.array(5.0, dtype=jnp.float32)

    def u(x):
        y = model(x)
        return y

    u_t = gradf(u, 0, 1)
    u_xx = gradf(u, 1, 2)

    res = u_t(collocs) - D*u_xx(collocs) - c*(u(collocs)-(u(collocs)**3))

    return res


# Burgers Equation
def burgers_res(model, collocs):
    
    # Eq. parameter
    nu = jnp.array(0.01/jnp.pi, dtype=float)

    def u(x):
        y = model(x)
        return y

    # Physics Loss Terms
    u_t = gradf(u, 0, 1)
    u_x = gradf(u, 1, 1)
    u_xx = gradf(u, 1, 2)

    # Get all residuals
    res = u_t(collocs) + u(collocs)*u_x(collocs) - nu*u_xx(collocs)

    return res


# Helmholtz Equation
def helmholtz_res(model, collocs):
    
    # Eq. parameters
    a1 = jnp.array(1.0, dtype=jnp.float32)
    a2 = jnp.array(4.0, dtype=jnp.float32)

    def u(x):
        y = model(x)
        return y

    u_xx = gradf(u, 0, 2)
    u_yy = gradf(u, 1, 2)

    # source term - we assume k = 1.0
    factor = 1.0 - (jnp.pi**2)*(a1**2 + a2**2)
    f = factor*jnp.sin(jnp.pi*a1*collocs[:,[0]])*jnp.sin(jnp.pi*a2*collocs[:,[1]])

    res = u_xx(collocs) + u_yy(collocs) + u(collocs) - f

    return res
    
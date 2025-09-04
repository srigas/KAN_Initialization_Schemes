import jax.numpy as jnp

def f1(x):
    return jnp.exp(-(x[:, [0]]**2) / (2 * x[:, [1]]**2)) / jnp.sqrt(2 * jnp.pi * x[:, [1]]**2)
    
def f2(x): 
    return jnp.exp(-((x[:, [0]] - x[:, [1]])**2) / (2 * x[:, [2]]**2)) / jnp.sqrt(2 * jnp.pi * x[:, [2]]**2)
    
def f3(x): 
    return 1 + x[:, [0]] * jnp.sin(x[:, [1]])
    
def f4(x): 
    return x[:, [0]] * (1 / x[:, [1]] - 1)
    
def f5(x): 
    return (x[:, [0]] + x[:, [1]]) / (1 + x[:, [0]] * x[:, [1]])
    
def f6(x): 
    return (1 + x[:, [0]] * x[:, [1]]) / (1 + x[:, [0]])
    
def f7(x): 
    return jnp.arcsin(x[:, [0]] * jnp.sin(x[:, [1]]))
    
def f8(x): 
    return 1 / (1 + x[:, [0]] * x[:, [1]])
    
def f9(x): 
    return jnp.sqrt(1 + x[:, [0]]**2 - 2 * x[:, [0]] * jnp.cos(x[:, [1]] - x[:, [2]]))
    
def f10(x): 
    return (jnp.sin(x[:, [0]] * x[:, [1]] / 2)**2) / (jnp.sin(x[:, [1]] / 2)**2)
    
def f11(x): 
    return x[:, [0]] * jnp.exp(-x[:, [1]])

def f12(x): 
    return jnp.cos(x[:, [0]]) + x[:, [1]] * jnp.cos(x[:, [0]])**2
    
def f13(x): 
    return (x[:, [0]] - 1) * x[:, [1]]
    
def f14(x): 
    return (1 / (4 * jnp.pi)) * x[:, [2]] * jnp.sqrt(x[:, [0]]**2 + x[:, [1]]**2)
    
def f15(x): 
    return x[:, [0]] * (1 + x[:, [1]] * jnp.cos(x[:, [2]]))
    
def f16(x):
    na = x[:, [0]] * x[:, [1]]
    return na / (1 - na / 3)
    
def f17(x): 
    return x[:, [0]] / (jnp.exp(x[:, [1]]) + jnp.exp(-x[:, [1]]))
    
def f18(x): 
    return x[:, [0]] + x[:, [1]] * x[:, [2]]
    
def f19(x): 
    return jnp.sqrt(1 + x[:, [0]]**2 + x[:, [1]]**2)
    
def f20(x): 
    return x[:, [1]] * (1 + x[:, [0]] * jnp.cos(x[:, [2]]))
    
import jax
import jax.numpy as jnp

from flax import nnx

from typing import Union, List

from jaxkan.layers.Spline import SplineLayer


class StdSplineLayer(SplineLayer):

    def _initialize_params(self, init_scheme, seed):

        key = jax.random.key(seed)

        # Also get distribution type
        distrib = init_scheme.get("distribution", "uniform")

        if distrib is None:
            distrib = "uniform"

        # Generate a sample of 10^5 points
        if distrib == "uniform":
            sample = jax.random.uniform(key, shape=(100000,), minval=-1.0, maxval=1.0)
        elif distrib == "normal":
            sample = jax.random.normal(key, shape=(100000,))

        # Finally get gain
        gain = init_scheme.get("gain", None)
        if gain is None:
            gain = sample.std().item()

        # ---- Residual Calculations --------
        # Variance equipartitioned across all terms
        scale = self.n_in * (self.grid.G + self.k + 1)
        # Apply the residual function
        y_res = self.residual(sample)
        # Calculate the average of residual^2(x)
        y_res_sq = y_res**2
        y_res_sq_mean = y_res_sq.mean().item()

        std_res = gain/jnp.sqrt(scale*y_res_sq_mean)
        c_res = nnx.initializers.normal(stddev=std_res)(self.rngs.params(), (self.n_out, self.n_in), jnp.float32)

        # ---- Basis Calculations -----------
        std_b = gain/jnp.sqrt(scale)
        c_basis = nnx.initializers.normal(stddev=std_b)(
            self.rngs.params(), (self.n_out, self.n_in, self.grid.G + self.k), jnp.float32
        )
        
        return c_res, c_basis

        
    def basis(self, x):
        basis_splines = super().basis(x)

        mean = jnp.mean(basis_splines, axis=0, keepdims=True)
        denom = jnp.sqrt(jnp.var(basis_splines, axis=0, keepdims=True) + 1e-5)
        basis_splines = (basis_splines - mean) / denom

        return basis_splines


class StdKAN(nnx.Module):
    
    def __init__(self, layer_dims: List[int], required_parameters: Union[None, dict] = None, seed: int = 42):
            
        if required_parameters is None:
            raise ValueError("required_parameters must be provided as a dictionary for the selected layer_type.")
        
        self.layers = [
                StdSplineLayer(
                    n_in=layer_dims[i],
                    n_out=layer_dims[i + 1],
                    **required_parameters,
                    seed=seed
                )
                for i in range(len(layer_dims) - 1)
            ]
    
    def __call__(self, x):

        # Pass through each layer of the KAN
        for layer in self.layers:
            x = layer(x)

        return x
        
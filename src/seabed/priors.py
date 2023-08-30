from jax import random
import jax.numpy as jnp

def uniform_prior_particles(key, minimums, maximums, N):
    """Produces an array of N particles sampled unformly within a 
    multidimensional box defined by minimums and maximums for each parameter.

    Parameters
    ----------
    key : jax.random.PRNGKey
        The pseudo-random number generator key used to seed all 
        jax random functions.
    minimums : Vector
        A vector of minimum parameter values
    maximums : Vector
        A vector of maximum parameter values
    N : Int
        The number of particles

    Returns
    -------
    Array
        A set of particles from the desired distribution. 
    """    
    n_params = len(minimums)
    return random.uniform(key,(n_params,N),
                          minval=jnp.asarray(minimums).reshape(n_params,1),
                          maxval=jnp.asarray(maximums).reshape(n_params,1))

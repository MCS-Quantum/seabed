import jax.numpy as jnp
from jax import jit, vmap

@jit
def diffable_plogp(p):
    """A automatically differentiable version of 
    the function ``p*log(p)`` for entropy calculations.

    Parameters
    ----------
    p : Float
        Probability

    Returns
    -------
    Float
        returns the value of ``p*log(p)``
    """    
    lp = jnp.log2(jnp.where(p>0,p,1))
    return lp*p

diffable_plogp_vec = jit(vmap(diffable_plogp,in_axes=(0,)))

@jit
def entropy_change(current_particles,current_weights,likelihoods):
    """Computes the difference in shannon entropy between posterior and prior
    particle distributions with the same particles but different weights.

    Parameters
    ----------
    current_particles : Array
        The array of particles. 
    current_weights : Array
        The current particle weights.
    likelihoods : Array
        An array of likelihoods which would be used to compute new
        particle weights of the prior distribution.

    Returns
    -------
    _type_
        _description_
    """    
    new_weights = current_weights*likelihoods
    new_weights = new_weights/jnp.sum(new_weights)
    H_old = -jnp.sum(diffable_plogp_vec(current_weights))
    H_new = -jnp.sum(diffable_plogp_vec(new_weights))
    return H_new-H_old
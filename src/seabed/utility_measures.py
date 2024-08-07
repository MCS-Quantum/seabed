# Copyright © 2023, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: SEABED
# By: Argonne National Laboratory
# BSD OPEN SOURCE LICENSE

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

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
def entropy(ws):
    return -jnp.sum(diffable_plogp_vec(ws))

@jit
def entropy_change(current_particles,current_weights,likelihoods):
    """Computes the difference in shannon entropy between posterior and prior
    particle distributions with the same particles but different weights.

    Mathematically, let P be the prior and Q be the posterior, then:
    entropy_change = H(Q) - H(P).

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
    Flot
        returns the change in entropy if bayesian updating were performed using 
        the given likelihoods.
    """    
    new_weights = current_weights*likelihoods
    new_weights = new_weights/jnp.sum(new_weights)
    H_old = entropy(current_weights)
    H_new = entropy(new_weights)
    return H_new - H_old

def entropy_based_utility(current_particles,current_weights,likelihoods):
    """Returns utility as -1 times the change in Shannon entropy for a
    given particle distribution.
    By maximizing utility we then minimize the posterior entropy.

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
    Flot
        returns the utility found by the expected change in Shannon entropy.
    """
    return -1 * entropy_change(current_particles,current_weights,likelihoods)

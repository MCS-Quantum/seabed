# Copyright © 2023, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: SEABED
# By: Argonne National Laboratory
# BSD OPEN SOURCE LICENSE

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

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

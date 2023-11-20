# Copyright © 2023, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: SEABED
# By: Argonne National Laboratory
# BSD OPEN SOURCE LICENSE

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

from jax import vmap,random,jit
import jax.numpy as jnp
from functools import partial

from .abstractbayesmodel import AbstractBayesianModel


class SimulatedModel(AbstractBayesianModel):
    """
    A type of BayesianModel where the likelihood
    function is derived from a computationally expensive
    simulation. 
    
    It is assumed that the computationally expensive component
    outputs an intermediate result that only depends on the 
    inputs and model parameters. Then this intermediate result 
    can be used to calculate likelihoods for certain outputs
    in a computationally inexpensive manner.
    
    The simulation_likelihood is a likelihood function that
    takes in input, output, parameters, and a precomputed_array.

    precompute_function(oneinput_vec,oneparameter_vec) --> precompute_data
    simulation_likelihood(oneinput_vec,oneoutput_vec,oneparameter_vec,precompute_data) --> likelihood
    
    """
    
    def __init__(self, key, particles, weights, expected_outputs,
                 precompute_function = None, 
                 multiparameter_precompute_function = None,
                 simulation_likelihood = None,
                 **kwargs):
        """Initialize an SimulatedModel object.

        Parameters
        ----------
        key : jax.random.PRNGKey
            The pseudo-random number generator key used to seed all 
            jax random functions.
        particles : Array
            The set of particles initializes to initialize the distribution.
            Has size ``n_dims``x``n_particles``. 
        weights : Array
            The set of weights for each particle. Has size (``n_particles``,). 
        expected_outputs : Array
            A set of possible outputs which may be observed.
        precompute_function : Function
            The precomputation function with function singature
              ``F(oneinput_vector,oneparameter_vector)``
            runs an expensive computation that will be passed
            into the simulation_likelihood function to calculate likelihoods. 
        multiparameter_precompute_function : Function
            The precomputation function with function singature
              ``F(oneinput_vector, multiple_parameter_vectors)``
            runs an expensive computation that will be passed
            into the simulation_likelihood function to calculate likelihoods.
        simulation_likelihood : Function
            A computationally inexpensive function with the signature
            ``G(oneinput_vector, oneoutput_vector, oneparameter_vector, precompute data)``
            that will compute the likelihoods of various outputs, by default None

        Raises
        ------
        ValueError
            An error is raised if no precompute function is passed. 
        """        
        
        self.lower_kwargs = kwargs
        self.simulation_likelihood = simulation_likelihood
        self.sim_likelihood_oneinput_oneoutput_multiparams = vmap(simulation_likelihood,in_axes=(None,None,-1,-1))
        self.sim_likelihood_oneinput_multioutput_multiparams = vmap(self.sim_likelihood_oneinput_oneoutput_multiparams,in_axes=(None,-1,None,None), out_axes=-1)
        self.sim_likelihood_oneinput_multioutput_oneparam = vmap(simulation_likelihood,in_axes=(None,-1,None,None))

        if multiparameter_precompute_function:

            def precompute_function(oneinput,oneparam):
                d = oneparam.shape[0]
                params = oneparam.reshape((d,1))
                pdata = multiparameter_precompute_function(oneinput,params)
                return pdata[...,0]
            
            self.precompute_function = precompute_function
            self.precompute_oneinput_multiparams = multiparameter_precompute_function

            
            def multiparameter_multioutput_likelihood_function(oneinput,multioutputs,multiparams):
                precomputes = multiparameter_precompute_function(oneinput,multiparams)
                
                return self.sim_likelihood_oneinput_multioutput_multiparams(oneinput,multioutputs,multiparams,precomputes)
            
            AbstractBayesianModel.__init__(self, key, particles, weights, expected_outputs,
                                        multiparameter_multioutput_likelihood_function=multiparameter_multioutput_likelihood_function
                                        ,**kwargs)

        elif precompute_function:
            self.precompute_function = precompute_function
            self.precompute_oneinput_multiparams = vmap(precompute_function,in_axes=(None,1),out_axes=(-1))

            
            def likelihood(oneinput_vec,oneoutput_vec,oneparameter_vec):
                precompute_data = precompute_function(oneinput_vec,oneparameter_vec)
                return simulation_likelihood(oneinput_vec,oneoutput_vec,oneparameter_vec,precompute_data)
            
            AbstractBayesianModel.__init__(self, key, particles, weights, expected_outputs,
                                        likelihood_function=likelihood,**kwargs)
        else:
            raise ValueError("No precompute function provided")
        
    def updated_weights_precomputes_from_experiment(self, oneinput, oneoutput, particles):
        """Returns the updated particle weights and precomputation data from a single
        input vector and output vector.

        Parameters
        ----------
        oneinput : Vector
            An input vector
        oneoutput : Vector
            An output vector
        particles :  Array
            An array of particles

        Returns
        -------
        Vector
            A vector of normalized particle weights.
        Array
            An array of precomputation data.
        """        
        precomputes = self.precompute_oneinput_multiparams(oneinput, particles)
        ls = self.sim_likelihood_oneinput_oneoutput_multiparams(oneinput, oneoutput, particles, precomputes)
        if jnp.any(jnp.isnan(ls)):
            raise ValueError("NaNs detected in likelihood calculation")
        weights = self.update_weights(ls)
        return weights, precomputes
        
    def updated_weights_from_precompute(self, oneinput, oneoutput, particles, precomputed_data):
        """Returns the updated particle weights using the most recently cached
        precompute data. 

        Parameters
        ----------
        oneinput : Vector
            An input vector
        oneoutput : Vector
            An output vector

        Returns
        -------
        Vector
            A vector of normalized particle weights.
        """        
        ls = self.sim_likelihood_oneinput_oneoutput_multiparams(oneinput, oneoutput, particles, precomputed_data)
        if jnp.any(jnp.isnan(ls)):
            raise ValueError("NaNs detected in likelihood calculation")
        weights = self.update_weights(ls)
        return weights
    
    def bayesian_update(self, oneinput, oneoutput):
        """Refines the parameter probability distribution function given an
        experimental input and output, resamples if needed, and updates
        the AbstractBayesianModel.
        """
        
        precomputed_data = self.precompute_oneinput_multiparams(oneinput,self.particles)
        
        self.bayesian_update_from_preompute(oneinput,oneoutput, precomputed_data)

    def bayesian_update_from_precompute(self, oneinput, oneoutput, precomputed_data):
        """Refines the parameter probability distribution function given an
        experimental input and output, resamples if needed, and updates
        the AbstractBayesianModel.

        Parameters
        ----------
        precomputed_data : Array
            An arrary 
        """
        
        self.weights = self.updated_weights_from_precompute(oneinput, oneoutput, self.particles, precomputed_data)
    
        if self.tuning_parameters['auto_resample']:
            self.resample_test()

    @partial(jit,static_argnames=['n_repeats'])
    def sample_output_kernel(self, key, oneinput, oneparam, n_repeats=1):
        pdata = self.precompute_function(oneinput,oneparam)
        ls = self.sim_likelihood_oneinput_multioutput_oneparam(oneinput,self.expected_outputs,oneparam,pdata)
        return random.choice(key,self.expected_outputs,shape=(n_repeats,),p=ls,axis=1)
    
    @partial(jit,static_argnames=['n_repeats'])
    def sample_outputs_kernel(self, keys, inputs, oneparam, n_repeats=1):
        f = jit(vmap(self.sample_output_kernel,in_axes=(0,1,None),out_axes=1))
        outputs = f(keys,inputs,oneparam,n_repeats=n_repeats)
        return outputs
    
    def sample_output(self, oneinput, oneparam, n_repeats=1):
        key, subkey = random.split(self.key)
        self.key = key
        output = self.sample_output_kernel(subkey,oneinput,oneparam,n_repeats=n_repeats)
        return output

    def sample_outputs(self, inputs, oneparam, n_repeats=1):
        """Samples from expected outputs of a process
        with multiple inputs and oneparameter.

        Useful for generating synthetic data. 

        Parameters
        ----------
        inputs : Array
            An array of input vectors. 
        oneparam : Vector
            A single parameter vector. 

        Returns
        -------
        Array
            An array of vectors of outputs sampled with the defined likelihood.  
        """        
        key, subkey = random.split(self.key)
        self.key = key
        subkeys = random.split(subkey,n_repeats)
        outputs = self.sample_outputs_kernel(subkeys,inputs,oneparam,n_repeats=n_repeats)
        return outputs

    def _tree_flatten(self):
        children = (self.key, self.particles, self.weights, self.expected_outputs)  # arrays / dynamic values
        aux_data = {'precompute_function':self.precompute_function, 
                    'simulation_likelihood':self.simulation_likelihood,
                    'multiparameter_precompute_function':self.precompute_oneinput_multiparams,
                    **self.lower_kwargs
                   }
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children,**aux_data)
    
from jax import tree_util
tree_util.register_pytree_node(SimulatedModel,
                               SimulatedModel._tree_flatten,
                               SimulatedModel._tree_unflatten)
import jax.numpy as jnp
from jax import vmap, random

from .particlepdf import ParticlePDF
from .utility_measures import entropy_change

class AbstractBayesianModel(ParticlePDF):
    """An abstract Bayesian probabilistic model for a system with 
    unknown parameters. This class defines a PraticlePDF model for a 
    system with oututs predictable from a likelihood function
    that is paramaterized by a set of underlying parameters 
    which are inteded to be estimated. 

    This abstract class implements the basic methods needed to 
    sequentially update the probabilistic model from new measurements
    and compute utilities of future experimental inputs.

    """

    def __init__(self, key, particles, weights, expected_outputs, 
                 likelihood_function=None, 
                 multiparameter_likelihood_function=None,
                 multiparameter_multioutput_likelihood_function=None,
                 utility_measure = entropy_change, 
                 **kwargs):
        """Initialize an AbstractBayesianModel object.

        There are three possible ways to define the likelihood function
        depending on the degree of parallelization you intend to utilize.

        If a single non-parallelized likelihood function is given
        vectorization with be performed with jax.vmap. 

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
        likelihood_function : Function, optional
            A function that computes likelihoods with the signature
            ``F(oneinput_vector, oneparameter_vector, oneoutput_vector)```
            , by default None
        multiparameter_likelihood_function : Function, optional
            A function that computes likelihoods for multiple parameters with the signature
            ``F(oneinput_vector, multiple_parameter_vectors, oneoutput_vector)```
            , by default None
        multiparameter_multioutput_likelihood_function : Function, optional
            A function that computes likelihoods for multiple parameters 
            and multiple outputs with the signature
            ``F(oneinput_vector, multiple_parameter_vectors, multiple_output_vectors)```
            , by default None
        utility_measure : Function, optional
            A measure of experimental utility with the signature
             ``U(current_particles,current_weights,likelihoods)``, by default entropy_change

        Raises
        ------
        ValueError
            Raises ValueError if no likelihood function is provided via 
            one of the three defined arguments.
        """        
        
        self.lower_kwargs = kwargs
        
        ParticlePDF.__init__(self, key, particles, weights, **kwargs)


        if likelihood_function:
            self.likelihood_function = likelihood_function # takes (oneinput_vec,oneoutput_vec,oneparameter_vec)
            self.oneinput_oneoutput_multiparams = vmap(likelihood_function,in_axes=(None,None,1)) # takes (oneinput_vec, oneoutput_vec, multi_param_vec)
            self.oneinput_multioutput_oneparam = vmap(likelihood_function,in_axes=(None,1,None))
            self.oneinput_multioutput_multiparams = vmap(self.oneinput_oneoutput_multiparams,in_axes = (None,1,None), out_axes=1)# takes (oneinput, multioutput, multiparameters)   


        elif multiparameter_likelihood_function:

            def likelihood_function(oneinput,oneoutput,oneparam):
                d = oneparam.shape[0]
                params = oneparam.reshape((d,1))
                ls = multiparameter_likelihood_function(oneinput,oneoutput,params)
                return ls[0]
            
            self.likelihood_function = likelihood_function # takes (oneinput_vec,oneoutput_vec,oneparameter_vec)
            self.oneinput_oneoutput_multiparams = multiparameter_likelihood_function
            self.oneinput_multioutput_oneparam = vmap(likelihood_function,in_axes=(None,1,None))
            self.oneinput_multioutput_multiparams = vmap(self.oneinput_oneoutput_multiparams,in_axes = (None,1,None), out_axes=1) # takes (oneinput, multioutput, multiparameters)   


        elif multiparameter_multioutput_likelihood_function:

            def likelihood_function(oneinput,oneoutput,oneparam):
                d = oneparam.shape[0]
                params = oneparam.reshape((d,1))
                b = oneoutput.shape[0]
                outputs = oneoutput.reshape((b,1))
                ls = multiparameter_multioutput_likelihood_function(oneinput,outputs,params)
                return ls[0,0]
            
            def oneinput_oneoutput_multiparams(oneinput,oneoutput,params):
                b = oneoutput.shape[0]
                outputs = oneoutput.reshape((b,1))
                ls = multiparameter_multioutput_likelihood_function(oneinput,outputs,params)
                return ls[:,0]
            
            def oneinput_multioutput_oneparam(oneinput,outputs,oneparam):
                d = oneparam.shape[0]
                params = oneparam.reshape((d,1))
                ls = multiparameter_multioutput_likelihood_function(oneinput,outputs,params)
                return ls[0,:]
            
            self.likelihood_function = likelihood_function # takes (oneinput_vec,oneoutput_vec,oneparameter_vec)
            self.oneinput_oneoutput_multiparams = oneinput_oneoutput_multiparams
            self.oneinput_multioutput_oneparam = oneinput_multioutput_oneparam
            self.oneinput_multioutput_multiparams = multiparameter_multioutput_likelihood_function

        else:
            raise ValueError("No likelihood function provided")


        self.utility_measure = utility_measure
        self.multioutput_utility = vmap(self.utility_measure,in_axes=(None,None,-1))
        self.expected_outputs = expected_outputs


    def updated_weights_from_experiment(self, oneinput, oneoutput):
        """Updates the particle weights of the AbstractBayesianModel
        for a single experimental run.

        Parameters
        ----------
        oneinput : Vector
            A vector of specified process inputs
        oneoutput : Vector
            A vector of observed process outputs

        Returns
        -------
        Vector
            The new, normalized particle weights.
        """        
        ls = self.oneinput_oneoutput_multiparams(oneinput, oneoutput, self.particles)
        if jnp.any(jnp.isnan(ls)):
            raise ValueError("NaNs detected in likelihood calculation")
        weights = self.update_weights(ls)
        return weights
    
    def bayesian_update(self, oneinput, oneoutput):
        """
        Refines the parameter probability distribution function given an
        experimental input and output, resamples if needed, and updates
        the AbstractBayesianModel.
        """
        self.weights = self.updated_weights_from_experiment(oneinput, oneoutput)
        
        if self.tuning_parameters['auto_resample']:
            self.resample_test()

    def batch_bayesian_update(self, inputset, outputset):
        """
        Refines the parameter probability distribution function given a set of
        experimental inputs and outputs, resamples as needed, and updates
        the AbstractBayesianModel.
        """
        n = inputset.shape[-1]
        for i in range(n):
            self.bayesian_update(inputset[:,i],outputset[:,i])
      
    def _expected_utility(self,oneinput,particles,weights):
        """Computes the expected value of utility of a single input based on a
        set of particles, weights, and utility measure. 

        Parameters
        ----------
        oneinput : Vector
            A single vector of inputs
        particles : Array
            An Array of particles over which to the expectation value will be taken
        weights : Vector
            A vector of particles specifying the weights of each particle. 

        Returns
        -------
        Float
            The expected utility of the input.
        """        
        # Compute a matrix of likelihoods for various output/parameter combinations. 
        ls = self.oneinput_multioutput_multiparams(oneinput,self.expected_outputs,particles) # shape n_particles x n_outputs
        # This gives the probability of measuring various outputs/parameters for a single input
        us = self.multioutput_utility(particles,weights,ls)

        return jnp.sum(jnp.dot(ls,us))
    
    
    def _expected_utilities(self,inputs,particles,weights):
        """Computes the expected value of utility of multiple inputs based on a set of
        particles, weights, and utility measure. 

        Parameters
        ----------
        inputs : Array
            An Array of inputs. 
        particles : Array
            An Array of particles over which to the expectation value will be taken
        weights : Vector
            A vector of particles specifying the weights of each particle. 

        Returns
        -------
        Vector
            The expected utility of each input.
        """        
        umap = vmap(self._expected_utility,in_axes=(1,None,None))
        return umap(inputs,particles,weights)
         
    
    def expected_utility(self,oneinput):
        """Computes the expected value of utility of a single input based on the object's
        current particles, weights, and utility measure. 

        Parameters
        ----------
        oneinput : Vector
            A single vector of inputs

        Returns
        -------
        Float
            The expected utility of the input.
        """        
        # Compute a matrix of likelihoods for various output/parameter combinations. 
        return self._expected_utility(oneinput,self.particles,self.weights)
    
    
    def expected_utilities(self,inputs):
        """Computes the expected value of utility of multiple inputs based the
        objects current particles, weights, and utility measure. 

        Parameters
        ----------
        inputs : Array
            An Array of inputs. 

        Returns
        -------
        Vector
            The expected utility of each input.
        """        
        return self._expected_utilities(inputs,self.particles,self.weights)

    def expected_utility_k_particles(self,oneinput,k=10):
        """Computes the expected value of utility of a single input using
         only the ``k`` particles with the largest weights. 

        Parameters
        ----------
        oneinput : Vector
            A single vector of inputs
        k : Int, optional
            A number of particles to use in the computation of the utility.

        Returns
        -------
        Float
            The expected utility of the input.
        """        
        # Compute a matrix of likelihoods for various output/parameter combinations. 
        key, subkey = random.split(self.key)
        self.key = key
        num_particles = self.particles.shape[1]
        I = random.choice(subkey,num_particles,shape=(k,),p=self.weights)
        particles = self.particles[:,I]
        weights = self.weights[I]
        return self._expected_utility(oneinput,particles,weights)
    
    def expected_utilities_k_particles(self,inputs,k=10):
        """Computes the expected value of utility of multiple inputs using
         only the ``k`` particles with the largest weights. 

        Parameters
        ----------
        inputs : Array
            An array of possible inputs.
        k : Int, optional
            A number of particles to use in the computation of the utility.

        Returns
        -------
        Float
            The expected utility of the input.
        """        
        key, subkey = random.split(self.key)
        self.key = key
        num_particles = self.particles.shape[1]
        I = random.choice(subkey,num_particles,shape=(k,),p=self.weights)
        particles = self.particles[:,I]
        weights = self.weights[I]
        return self._expected_utilities(inputs,particles,weights)
    
    def sample_output(self,oneinput,oneparam):
        """Samples from expected outputs of a process
        with oneinput and oneparameter.

        Useful for generating a synthetic datum. 

        Parameters
        ----------
        oneinput : Vector
            A single input vector. 
        oneparam : Vector
            A single parameter vector. 

        Returns
        -------
        Vector
            A single vector of outputs sampled with the defined likelihood.  
        """        
        # This can definitely be re-written to parallelize the computation of likelihoods up-front 
        # but creating a synthetic dataset doesn't really need to be fast at the moment.
        ls = self.oneinput_multioutput_oneparam(oneinput,self.expected_outputs,oneparam)
        key, subkey = random.split(self.key)
        self.key = key
        output = random.choice(subkey,self.expected_outputs,p=ls,axis=1)
        return output
    
    def sample_outputs(self, inputs, oneparam):
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
        # This can definitely be re-written to parallelize the computation of likelihoods up-front 
        # but creating a synthetic dataset doesn't really need to be fast at the moment.
        num_inputs = inputs.shape[1]
        return jnp.stack([self.sample_output(inputs[:,i],oneparam) for i in range(num_inputs)],axis=1)
        
    def _tree_flatten(self):
        children = (self.key, self.particles, self.weights, self.expected_outputs)  # arrays / dynamic values
        aux_data = {'likelihood_function':self.likelihood_function,
                    'multiparameter_likelihood_function':self.oneinput_oneoutput_multiparams,
                    'multiparameter_multioutput_likelihood_function':self.oneinput_multioutput_multiparams,
                    'utility_measure':self.utility_measure, **self.lower_kwargs}
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children,**aux_data)
    
from jax import tree_util
tree_util.register_pytree_node(AbstractBayesianModel,
                               AbstractBayesianModel._tree_flatten,
                               AbstractBayesianModel._tree_unflatten)


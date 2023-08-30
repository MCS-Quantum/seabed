from jax import vmap

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
            The latest precomputation results are cached for future re-use 
            to minimize computational overhead, by default None
        multiparameter_precompute_function : Function
            The precomputation function with function singature
              ``F(oneinput_vector, multiple_parameter_vectors)``
            runs an expensive computation that will be passed
            into the simulation_likelihood function to calculate likelihoods. 
            The latest precomputation results are cached for future re-use 
            to minimize computational overhead, by default None
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
        self.latest_precomputed_data = None

        if precompute_function:
            self.precompute_function = precompute_function
            self.precompute_oneinput_multiparams = vmap(precompute_function,in_axes=(None,1),out_axes=(-1))

            
            def likelihood(oneinput_vec,oneoutput_vec,oneparameter_vec):
                precompute_data = precompute_function(oneinput_vec,oneparameter_vec)
                return simulation_likelihood(oneinput_vec,oneoutput_vec,oneparameter_vec,precompute_data)
            
            AbstractBayesianModel.__init__(self, key, particles, weights, expected_outputs,
                                        likelihood_function=likelihood,**kwargs)

        elif multiparameter_precompute_function:

            def precompute_function(oneinput,oneparam):
                d = oneparam.shape[0]
                params = oneparam.reshape((d,1))
                ls = multiparameter_precompute_function(oneinput,params)
                return ls[0]
            
            self.precompute_function = precompute_function
            self.precompute_oneinput_multiparams = multiparameter_precompute_function

            
            def multiparameter_multioutput_likelihood_function(oneinput,multioutputs,multiparams):
                precomputes = multiparameter_precompute_function(oneinput,multiparams)
                
                return self.sim_likelihood_oneinput_multioutput_multiparams(oneinput,multioutputs,multiparams,precomputes)
            
            AbstractBayesianModel.__init__(self, key, particles, weights, expected_outputs,
                                        multiparameter_multioutput_likelihood_function=multiparameter_multioutput_likelihood_function
                                        ,**kwargs)


        else:
            raise ValueError("No precompute function provided")
        
    
    def updated_weights_precomputes_from_experiment(self, oneinput, oneoutput):
        """Returns the updated particle weights and precomputation data from a single
        input vector and output vector.

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
        Array
            An array of precomputation data.
        """        
        precomputes = self.precompute_oneinput_multiparams(oneinput,self.particles)
        ls = self.sim_likelihood_oneinput_oneoutput_multiparams(oneinput, oneoutput, self.particles, precomputes)
        weights = self.update_weights(ls)
        return weights, precomputes
        
    
    def updated_weights_from_precompute(self, oneinput, oneoutput):
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
        ls = self.sim_likelihood_oneinput_oneoutput_multiparams(oneinput, oneoutput, self.particles, self.latest_precomputed_data)
        weights = self.update_weights(ls)
        return weights
        
    def bayesian_update(self, oneinput, oneoutput, use_latest_precompute=False):
        """Refines the parameter probability distribution function given an
        experimental input and output, resamples if needed, and updates
        the AbstractBayesianModel.

        Parameters
        ----------
        use_latest_precompute : bool, optional
            Specifies whether or not the most recently cached precompute data
            is used for likelihood computation, by default False.
            If use_latest_precompute=True and the ParticlePDF hasn't been resampled
            then the previously cached precompute results are input into the 
            simulation_likelihood function.
        """
        
        if use_latest_precompute and not self.just_resampled:
            self.weights = self.updated_weights_from_precompute(oneinput, oneoutput)
        else:
            self.weights, self.latest_precomputed_data = self.updated_weights_precomputes_from_experiment(oneinput, oneoutput)
        
        if self.tuning_parameters['auto_resample']:
            self.resample_test()

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
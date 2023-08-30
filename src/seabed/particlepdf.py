import jax.numpy as jnp
from jax import random, jit
from .samplers import sample_particles, Liu_West_resampler


class ParticlePDF:
    """
    A probability distribution class defined by a finite set of 
    discrete 'particles' in arbitrary dimensions. Each particle
    has an associated weight corresponding with the probability.
    This class has methods associated with the updating of the 
    particle weights from likelihoods and observed data, sampling
    from the distribution and 'resampling' the distribution. 

    Warnings:
        The number of samples (i.e. particles) required for good performance
        will depend on the application.  Too many samples will slow down the
        calculations, but too few samples can produce incorrect results.
    """

    def __init__(self, key, particles, weights, 
                 resampler = Liu_West_resampler,
                 tuning_parameters = {'resample_threshold':0.5,'auto_resample':True},
                 resampling_parameters = {'a':0.98, 'scale':True}, 
                 just_resampled=False, **kwargs):
        """Initialize a ParticlePDF object. 

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
        resampler : Function, optional
            A function that resamples the ParticlePDF, by default Liu_West_resampler.
        tuning_parameters : dict, optional
            The parameters that determine how the ParticlePDF updates
            with new data, by default
            {'resample_threshold':0.5,'auto_resample':True}.
        resampling_parameters : dict, optional
            The parameters passed to the resampler function, by default
            {'a':0.98, 'scale':True}.
        just_resampled : bool, optional
            Specifies if the PDF has been recently resampled, by default False.
        """        
        
        # The jax.random.PRNGkey() for random number sampling
        self.key = key
        
        self.particles = particles

        self.weights = weights
        
        self.n_particles = self.particles.shape[1]

        self.n_dims = self.particles.shape[0]
        
        self.resampler=resampler

        self.tuning_parameters = tuning_parameters
        self.resampling_parameters = resampling_parameters

        self.just_resampled = just_resampled

    @jit
    def mean(self):
        """Calculates the mean of the probability distribution.

        Returns
        -------
        Float
            The weighted mean of the parameter distribution.
        """        
        return jnp.average(self.particles, axis=1,
                          weights=self.weights)
    @jit
    def covariance(self):
        """Calculates the covariance matrix of the probability distribution.

        Returns
        -------
        Array
            The covariance of the parameter distribution as an
            ``n_dims`` X ``n_dims`` array.
        """
        n_dims = self.particles.shape[0]
        raw_covariance = jnp.cov(self.particles, aweights=self.weights)
        if n_dims == 1:
            return raw_covariance.reshape((1, 1))
        else:
            return raw_covariance

    @jit
    def update_weights(self, likelihoods):
        """Performs a update on the probability distribution weights
        based on Baye's rule and normalizes the new weights.

        Parameters
        ----------
        likelihoods : Array
            An array of likelihoods for each particle. 

        Returns
        -------
        Array
            The new normalized weights. 
        """
        weights = (likelihoods*self.weights)
        return weights/jnp.sum(weights)
        

    def bayesian_update(self, likelihoods):
        """Updates the ParticlePDF weights with a Bayesian update 
        and performs resampling if needed.

        Parameters
        ----------
        likelihoods : Array
            An array of likelihoods for each particle. 

         """
        self.weights = self.update_weights(likelihoods)
        
        if self.tuning_parameters['auto_resample']:
            self.resample_test()
    
    @jit
    def n_eff(self):
        """Calculates the number of effective particles.

        Returns
        -------
        Float
            The number of `effective` particles. 
        """        
        wsquared = jnp.square(self.weights)
        return 1 / jnp.sum(wsquared)
    
    def resample_test(self):        
        """Tests the distribution and performs a resampling if required.

        If the effective number of particles falls below
        ``self.tuning_parameters['resample_threshold'] * n_particles``,
        performs a resampling.  Sets ``just_resampled`` to ``True``.
        """
        threshold = self.tuning_parameters['resample_threshold']
        n_eff = self.n_eff()
        if n_eff / self.n_particles < threshold:
            key, subkey = random.split(self.key)
            self.particles, self.weights = self.resample(subkey)
            self.key = key
            self.just_resampled = True
        else:
            self.just_resampled = False
        
    @jit
    def resample(self,key):
        """Performs a resampling of the distribution as specified by 
        self.resampler and self.resampler_params.

        Resampling provides numerical stability to Sequential Monte Carlo
        methods based on Bayseian updates and particle filters.

        As updates to the distribution are made some particles develop very
        low weights because they are highly unlikely. Resampling draws new
        particles to ensure that regions of high probability parameter space 
        are effectively populated throughout the learning process. 

        Returns
        -------
        Array
            Newly sample particles
        
        Array
            New particle weights
        """        

        # Call the resampler function to get a new set of particles
        new_particles = self.resampler(key, self.particles, self.weights, **self.resampling_parameters)
        # Re-fill the current particle weights with 1/n_particles
        new_weights = jnp.full(self.n_particles, 1/self.n_particles)
        return new_particles, new_weights

    def randdraw(self, n_draws=1):
        """Provides random parameter draws from the distribution

        Particles are selected randomly with probabilities given by
        ``self.weights``.

        Parameters
        ----------
        n_draws : int, optional
            The number of draws requested, by default 1

        Returns
        -------
        Array
            ``n_dims`` x ``n_draws`` array of particles. 
        """
        key, subkey = random.split(self.key)
        self.key = key
        return sample_particles(subkey, self.particles, self.weights, n=n_draws)
    
    
    def _tree_flatten(self):
        children = (self.key, self.particles, self.weights)  # arrays / dynamic values
        aux_data = {'resampler':self.resampler,
                    'tuning_parameters': self.tuning_parameters,
                    'resampling_parameters':self.resampling_parameters,
                    'just_resampled':self.just_resampled}  # static values
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children,**aux_data)


from jax import tree_util
tree_util.register_pytree_node(ParticlePDF,
                               ParticlePDF._tree_flatten,
                               ParticlePDF._tree_unflatten)
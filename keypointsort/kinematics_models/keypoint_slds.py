import jax
import jax.numpy as jnp
import jax.random as jr
from keypoint_moseq.util import *
from keypoint_moseq.gibbs import *
from keypoint_moseq.initialize import *
na = jnp.newaxis


def isotropic_gaussian_log_prob(y, mu, sigmasq):
    diff = ((y-mu)**2).sum(-1)
    return -diff/2/sigmasq -jnp.log(sigmasq)/2*y.shape[-1] 


def simple_kalman_filter(mask, ms, ss, m0, s0, q):
    
    def _step(carry, args):
        ll, m_pred, s_pred, m, s = *carry, *args
        ll += isotropic_gaussian_log_prob(m, m_pred, s_pred + s)
        s_cond = 1/(1/s_pred + 1/s)
        m_cond = s_cond*(m_pred/s_pred + m/s)
        return (ll, m_cond, s_cond + q), (m_cond, s_cond)
    
    def _masked_step(carry, args):
        return carry, carry[1:]

    (ll, _, _), (filtered_ms, filtered_ss) = jax.lax.scan(
        lambda carry,args: jax.lax.cond(
            args[0]>0, _step, _masked_step, carry, args[1:]),
        (0., m0, s0), (mask, ms, ss))
    
    return ll, filtered_ms, filtered_ss


def simple_kalman_sample(key, mask, ms, ss, m0, s0, q):
    
    def _step(xn, args):
        weight, sample = args
        x = weight*xn + sample
        return x, x
    
    def _masked_step(x, args):
        return x,jnp.zeros_like(x)

    ll, filtered_ms, filtered_ss = simple_kalman_filter(mask, ms, ss, m0, s0, q)
    s_conds = 1/(1/filtered_ss + 1/q)
    samples = jr.normal(key, filtered_ms.shape) * jnp.sqrt(s_conds)[:,na]
    samples += (s_conds/filtered_ss)[:,na] * filtered_ms
    xT = jr.normal(key, filtered_ms[-1].shape)
    xT = xT*jnp.sqrt(filtered_ss[-1]) + filtered_ms[-1]
    
    args = (mask[:-1], (s_conds/q)[:-1], samples[:-1])
    _, xs = jax.lax.scan(lambda carry,args: jax.lax.cond(
        args[0]>0, _step, _masked_step, carry, args[1:]), xT, args, reverse=True)
    return jnp.vstack([xs, xT])


'''
@jax.jit
def sample_positions(key, obs, mask, outliers, hypparams, *, x, s, v, h, Cd, sigmasq, **kwargs):
    k,d = obs.shape[-2:]
    Gamma = center_embedding(k)
    Ybar = Gamma @ (pad_affine(x)@Cd.T).reshape(*obs.shape[:-2],k-1,d)
    model_mean = affine_transform(Ybar, v, h)
    model_var = s * sigmasq

    obs_variance = (hypparams['outlier_variance']*outliers
                   +hypparams['inlier_variance']*(1-outliers))
    ss = 1/(1/obs_variance + 1/model_var)
    ms = (obs/obs_variance[:,:,:,na] + model_mean/model_var[:,:,:,na]) * ss[:,:,:,na]
 
    keys = jr.split(key, obs.shape[-2])
    q = hypparams['keypoint_dx_variance']
    resample_one = jax.vmap(simple_kalman_sample, in_axes=(0,na,-2,-1,-2,na,na), out_axes=-2)
    resample_all = jax.vmap(resample_one,  in_axes=(na,0,0,0,0,na,na), out_axes=0)
    return resample_all(keys, mask, ms, ss, ms[:,0], 1., q)
'''    
    
def position_potentials(obs, outliers, hypparams, *, x, s, v, h, Cd, sigmasq, **kwargs):
    k,d = obs.shape[-2:]
    Gamma = center_embedding(k)
    Ybar = Gamma @ (pad_affine(x)@Cd.T).reshape(*obs.shape[:-2],k-1,d)
    model_mean = affine_transform(Ybar, v, h)
    model_var = s * sigmasq
    obs_variance = (hypparams['outlier_variance']*outliers
                   +hypparams['inlier_variance']*(1-outliers))
    ss = 1/(1/obs_variance + 1/model_var)
    ms = (obs/obs_variance[:,:,:,na] + model_mean/model_var[:,:,:,na]) * ss[:,:,:,na]
    return ms, ss
  

@jax.jit
def sample_positions(key, obs, mask, outliers, hypparams, **kwargs):
    ms, ss = position_potentials(obs, outliers, hypparams, **kwargs)
    keys = jr.split(key, obs.shape[-2])
    q = hypparams['keypoint_dx_variance']
    resample_one = jax.vmap(simple_kalman_sample, in_axes=(0,na,-2,-1,-2,na,na), out_axes=-2)
    resample_all = jax.vmap(resample_one,  in_axes=(na,0,0,0,0,na,na), out_axes=0)
    return resample_all(keys, mask, ms, ss, ms[:,0], 1., q)
 
@jax.jit
def filter_positions(obs, mask, outliers, hypparams, **kwargs):
    ms, ss = position_potentials(obs, outliers, hypparams, **kwargs)
    q = hypparams['keypoint_dx_variance']
    filter_fn = lambda *args: simple_kalman_filter(*args)[1]
    filter_one = jax.vmap(filter_fn, in_axes=(na,-2,-1,-2,na,na), out_axes=-2)
    filter_all = jax.vmap(filter_one,  in_axes=(0,0,0,0,na,na), out_axes=0)
    return filter_all(mask, ms, ss, ms[:,0], 1., q)


@jax.jit
def sample_outliers(key, obs, positions, outlier_prob, hypparams):
    outlier_lp = observations_log_prob(
        obs, positions, hypparams['outlier_variance']
    ) + jnp.log(outlier_prob)
    
    inlier_lp = observations_log_prob(
        obs, positions, hypparams['inlier_variance']
    ) + jnp.log(1-outlier_prob)
    
    prob = jnp.exp(outlier_lp - jnp.logaddexp(outlier_lp,inlier_lp))
    return jr.bernoulli(key, prob)

@jax.jit
def resample_outliers(key, obs, outlier_prob, *, y, outlier_variance, inlier_variance, **kwargs):
    outlier_lp = isotropic_gaussian_log_prob(obs, y, outlier_variance) * jnp.log(outlier_prob)
    inlier_lp = isotropic_gaussian_log_prob(obs, y, inlier_variance) * jnp.log(1-outlier_prob)
    prob = jnp.exp(outlier_lp - jnp.logaddexp(outlier_lp,inlier_lp))
    return jr.bernoulli(key, prob)

   
@jax.jit
def observations_log_prob(obs, positions, variance):
    sqdev = ((obs-positions)**2).sum(-1)
    norm = jnp.log(variance)/2*obs.shape[-1]
    return -sqdev/2/variance-norm


class KeypointSLDS:
    def __init__(self, hypparams, params=None):
        self.key = jr.PRNGKey(0)
        self.hypparams = hypparams
        if params is None: self.params = {}
        else: self.params = params
        self.states = {}
        self.positions = None
        self.outliers = None
    
    def initialize(self, obs, outlier_prob, mask):
        obs = jnp.squeeze(obs, axis=3)
        outlier_prob = jnp.squeeze(outlier_prob, axis=3)
        self.key, *keys = jr.split(self.key, 6)
        data = {'Y':obs, 'mask':mask}
    
        # initialize outliers
        self.outliers = jr.bernoulli(keys[0], outlier_prob)
        
        # initialize centroid/heading
        self.states['v'] = v = initial_location(
            **data, **self.states, outliers=self.outliers)

        self.states['h'] = h = initial_heading(
            **data, **self.states, **self.hypparams, outliers=self.outliers)
        
        # initialize latent states and obs params
        if not 'sigmasq' in self.params: 
            self.params['sigmasq'] = jnp.ones(data['Y'].shape[-2]) * \
                self.hypparams['obs_hypparams']['sigmasq_0']

        self.states['s'] = jnp.ones(obs.shape[:-1]) * \
            self.hypparams['obs_hypparams']['s_0']

        if not 'Cd' in self.params:
            self.states['x'],self.params['Cd'] = initial_latents(
                keys[1], latent_dim=self.hypparams['latent_dim'], 
                **data, **self.states, outliers=self.outliers)
        else: 
            Cd,Gamma = self.params['Cd'], center_embedding(data['Y'].shape[-2])
            ys = inverse_affine_transform(data['Y'],v,h).reshape(*data['Y'].shape[:-2],-1)
            self.states['x'] = (ys@jnp.kron(Gamma, jnp.eye(2)) - Cd[:,-1]) @ Cd[:,:-1]

        # initialize positions
        self.positions = sample_positions(
            keys[2], obs, mask, self.outliers, self.hypparams, **self.states, **self.params)

        # initialize AR params and stateseq
        if not 'pi' in self.params: self.params['betas'],self.params['pi'] = \
            initial_hdp_transitions(keys[3], **self.hypparams['trans_hypparams'])
        if not 'Ab' in self.params: self.params['Ab'],self.params['Q'] = \
            initial_ar_params(keys[4], **self.hypparams['ar_hypparams'])
        self.states['z'] = resample_stateseqs(keys[4], **data, **self.states, **self.params)

    
    def resample_states(self, data):
        self.key, *keys = jr.split(self.key, 5)

        self.states['z'] = resample_stateseqs(
            keys[0], **data, **self.states, **self.params)

        self.states['x'] = resample_latents(
            keys[1], **data, **self.states, **self.params)

        self.states['h'] = resample_heading(
            keys[2], **data, **self.states, **self.params)

        self.states['v'] = resample_location(
            keys[3], **data, **self.states, **self.params, 
            **self.hypparams['translation_hypparams'])
        
    def resample_params(self, data):
        self.key, *keys = jr.split(self.key, 4)
        
        self.params['Ab'],self.params['Q'] = resample_ar_params(
            keys[0], **data, **self.states, **self.params, 
            **self.hypparams['ar_hypparams'])

        self.params['Cd'] = resample_obs_params(
            keys[1], **data, **self.states, **self.params, 
            **self.hypparams['obs_hypparams'])
        
        self.params['betas'],self.params['pi'] = resample_hdp_transitions(
            keys[2], **data, **self.states, **self.params, 
            **self.hypparams['trans_hypparams'])
        
        
    def step(self, obs, outlier_prob, mask, update_params=False):
        self.key, *keys = jr.split(self.key, 4)
        
        obs = jnp.squeeze(obs, axis=3)
        outlier_prob = jnp.squeeze(outlier_prob, axis=3)

        self.outliers = sample_outliers(
            keys[0], obs, self.positions, outlier_prob, self.hypparams)

        self.positions = sample_positions(
            keys[2], obs, mask, self.outliers, self.hypparams, **self.states, **self.params)

        data = {'Y': self.positions, 'mask':mask}
        self.resample_states(data)
        if update_params: self.resample_params(data)



    def obs_log_likelihood(self, obs, outlier_prob):
        obs = jnp.squeeze(obs, axis=3)
        outlier_prob = jnp.squeeze(outlier_prob, axis=3)
        
        outlier_lp = observations_log_prob(
            obs, self.positions, self.hypparams['outlier_variance'])
        
        inlier_lp = observations_log_prob(
            obs, self.positions, self.hypparams['inlier_variance'])
        
        prior = jnp.stack([1-outlier_prob, outlier_prob])
        lp = jax.nn.logsumexp(jnp.stack([inlier_lp,outlier_lp]), b=prior, axis=0)
        return lp[...,na]
    
    
    def filtered_positions(self, obs, mask):
        obs = jnp.squeeze(obs, axis=3)
        return filter_positions(
            obs, mask, self.outliers, self.hypparams, **self.states, **self.params)

        


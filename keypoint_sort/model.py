import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial
from itertools import permutations
import gimbal
na = jnp.newaxis

from .util import log_normalize, gaussian_log_prob


#-----------------------------------------------------------------------------#
#                              Keypoint sorting model                         #
#-----------------------------------------------------------------------------#

def viterbi_assignments(
    parents, assign_lls,  
    clique_affinity_row_norm,
    clique_affinity_col_norm):
    """
    Compute max likelihood assignments for each keypoint observation
    given affinity scores and independent keypoint/assignment log-likelihoods.
    """
    # get subtree potentials by message passing up the tree
    N,T,K,C = assign_lls.shape[1:]
    permutations = all_permutations(N)
    clique_marg, log_norm = log_normalize(
        (assign_lls[permutations] * jnp.eye(N).reshape(1,N,N,1,1,1)).sum((1,2)))
    
    # get marginals going up the tree
    pass_backward = partial(
        tree_pass_backward, parents, 
        clique_affinity_col_norm)
    subtree_marg = jax.lax.scan(pass_backward, clique_marg, jnp.arange(K,0,-1))[0]

    # get ml assignments going down the tree
    pass_forward = partial(
        tree_max_likelihood_forward, parents,
        subtree_marg, clique_affinity_row_norm)
    samples = jax.lax.scan(pass_forward, jnp.zeros((T,K,C), dtype=int), jnp.arange(K,0,-1))[0]
    return jnp.transpose(permutations[samples], axes=(3,0,1,2))


@jax.jit
def initial_assignments(
    observations, identity, 
    clique_affinity_row_norm,
    clique_affinity_col_norm,
    parents, eta):
    """
    To initialize assignments, first get maximum likelihood assignments
    for each frame separately using just affinities and identity probabilities,
    then use the continuity of keypoint locations to optionally permute
    the labels in each frame (eta = gaussian kernel variance for continuity).
    """
    N, T, K, C = observations.shape[:-1]
    permutations = all_permutations(N)

    assignments_init = viterbi_assignments(
        parents, identity,
        clique_affinity_row_norm,
        clique_affinity_col_norm)

    # get permutation_lls[p,t,c] = log P (identity | permutation p (assignments_init[time t and camera c]))
    group_lls = split_animals(jnp.moveaxis(identity,1,-1), assignments_init).sum(2)
    permutations_lls = (group_lls[permutations] * jnp.eye(N).reshape(1,N,1,1,N)).sum((1,4))

    # tmats[c,t,:,:] = permutation->permutation transitions probs, based on 
    # random walk model with variance eta for keypoint motion across frames
    obs_split = split_animals(observations, assignments_init)
    permutation_transition_ll = -(jnp.nan_to_num(obs_split[permutations,1:] - obs_split[na,:,:-1])**2).sum((1,3,5)) / eta
    perm2perm = jnp.argmax((permutations[:,permutations][:,:,na,:]==permutations[na,na:,:]).sum(-1),-1) 
    transition_lls = permutation_transition_ll[perm2perm,:,:]
    tmats = jnp.exp(transition_lls - transition_lls.max(1, keepdims=True))+1e-6
    log_tmats = jnp.log(jnp.transpose(tmats / tmats.sum(1, keepdims=True), axes=(3,2,0,1)))

    # apply simple hmm viterbi algo
    L = log_tmats + permutations_lls.T[:,1:,na,:]
    P0 = jnp.ones((C,L.shape[2]))/L.shape[2]
    seqs = jax.vmap(viterbi_simple)(L,P0)

    # apply permutations to `assignments_init`
    assignments_init = jnp.take_along_axis(assignments_init, permutations[seqs].T.reshape(N,T,1,C), axis=0)
    return assignments_init



def viterbi_simple(L, P0):
    """Vanilla Viterbi algorithm where L[t,i,j] = log P( seq[t+1]=j | seq[t]=i )
    """
    def forward(trellis, L):
        P = trellis[:,na] + L
        trellis = P.max(0)
        pointer = jnp.argmax(P,axis=0)
        return trellis, pointer

    def back(i,pointer):
        return (pointer[i],pointer[i])
    trellis,pointers = jax.lax.scan(forward, P0, L)
    end_state = jnp.argmax(trellis,axis=0)
    seq = jax.lax.scan(back, end_state, pointers, reverse=True)[1]
    return jnp.append(seq, end_state)


def all_permutations(N):
    return jnp.array(list(permutations(range(N))))

def tree_pass_backward(parents, clique_affinity_col_norm, subtree_marg, k):
    msg = jax.nn.logsumexp(clique_affinity_col_norm[:,:,:,k,:]+subtree_marg[:,na,:,k,:],axis=0)
    return subtree_marg.at[:,:,parents[k],:].add(msg), None

def tree_samp_forward(parents, subtree_marg, clique_affinity_row_norm, carry, k):
    key, samples = carry
    key, newkey = jr.split(key)
    cond_k = jnp.take_along_axis(clique_affinity_row_norm[:,:,:,k], samples[:,parents[k]][na,na], axis=1)[:,0]
    marg_k = subtree_marg[:,:,k,:] + cond_k
    sample = jr.categorical(newkey, marg_k, axis=0)
    return (key, samples.at[:,k].set(sample)), None

def split_animals(X, assignments):
    """Use assignments to sort an array X of shape (B,N,T,K,C,...) such 
    that all keypoints for animal i are in slice [:,i,:,:,:,...]
    """
    extra_axes = len(X.shape)-len(assignments.shape)
    assignments = assignments.reshape(*assignments.shape, *[1]*extra_axes)
    return jnp.take_along_axis(X, assignments, axis=0)

@jax.jit
def forward_backward_assignments(
    key, parents, assign_lls, 
    clique_affinity_row_norm,
    clique_affinity_col_norm):
    """
    Sample posterior assignments for each keypoint observation
    given affinity scores and independent keypoint/assignment log-likelihoods.
    """
    # get subtree potentials by message passing up the tree
    N,T,K,C = assign_lls.shape[1:]
    permutations = all_permutations(N)
    clique_marg, log_norm = log_normalize(
        (assign_lls[permutations] * jnp.eye(N).reshape(1,N,N,1,1,1)).sum((1,2)))

    # get marginals going up the tree
    pass_backward = partial(
        tree_pass_backward, parents, 
        clique_affinity_col_norm)
    subtree_marg = jax.lax.scan(pass_backward, clique_marg, jnp.arange(K,0,-1))[0]
    
    # sample full posterior sample by sampling down the tree
    pass_forward = partial(
        tree_samp_forward, parents, 
        subtree_marg, clique_affinity_row_norm)
    samples = jax.lax.scan(pass_forward, (key, jnp.zeros((T,K,C),dtype=int)), jnp.arange(K,0,-1))[0][1]
    
    log_likelihood = (
          jnp.nan_to_num(log_norm).sum()
        + jnp.nan_to_num(jnp.take_along_axis(
            clique_marg, samples[na], axis=0)).sum()
        + jnp.nan_to_num(jnp.take_along_axis(
            jnp.take_along_axis(
                clique_affinity_row_norm, samples[na,na], axis=0),
            samples[na,na,:,parents], axis=1)).sum())
    
    return jnp.transpose(permutations[samples], axes=(3,0,1,2)), log_likelihood


def tree_max_likelihood_forward(parents, subtree_marg, clique_affinity_row_norm, samples, k): 
    cond_k = clique_affinity_row_norm[:,samples[:,parents[k],:],
                                       jnp.arange(samples.shape[0])[:,na],k,
                                       jnp.arange(samples.shape[2])[na,:]]
    sample = jnp.argmax(subtree_marg[:,:,k,:] + cond_k, axis=0)
    return samples.at[:,k,:].set(sample), None



class KeypointSort:
    """
    Keypoint sorting model.
    
    Given keypoint observations for multiple animals, the goal
    is to infer the animal-identity for each keypoint. Inference 
    is based on part-affinity-field scores for the keypoints and 
    on a kinematics model for each animal. The kinematics model
    is modular: any model can be used as long as it can produce
    a log-likelihood score for each keypoint observation. 

    Below, `N` is the number of animals, `T` is the number of
    time steps, `K` is the number of keypoints, and `C` is the 
    number of cameras.
    
    Parameters
    ----------
    parents : ndarray, shape (K,)
        The parent of each joint. parent[i] = [parent of node i]. The root 
        node should be its own parent.

    kinematics_models : list of objects (one per animal)
        Objects to use for kinematics modeling. Must implement:
     
        - `initialize(observations, outlier_probs) -> None`
            Initializes model based on observations for a single 
            animal (using initial identity guesses).

        - `step(observations, outlier_probs) -> None`
            Updates latent states of the model given observations
            for a single animal (using current identity assignments).

        - `obs_log_likelihood(observations, outlier_probs) -> log_likelihood`
            Calculates log likelihood of observations (all instances 
            of each keyoint) given the current model parameters; used
            to resample identity assignments.

    observations: ndarray, shape (N,T,K,C,2)
        Location of instance i of keypoint k for time t and camera c
        
    outlier_probs: ndarray, shape (N,T,K,C)
        Outlier probabilities based on neural net confidence

    affinity : ndarray, shape (N,N,T,K,C)
        Log probability that two keypoints observations belong to same animal.
        `affinity[i,j,t,k,c]` represents the log probability that  instance i 
        of keypoint k is from the same animal as instance j of keypoint parent(k) 
        for time t and camera c.

    identity : ndarray, shape (N,N,T,K,C)
        Log probability that each keypoint observation belongs to a specific
        animal. `identity[i,j,t,k,c]` represents the log probability that 
        instance i of keypoint k belongs to animal j for time t and camera c.

    Attributes
    ----------
    assignments: ndarray, shape (N,T,K,C)
        The current identity assignment for each keypoint observation.
        `assignment[i,t,k,c]` is the assignment of instance i of keypoint
        k for time t and camera c.  
    
    permutations : ndarray, shape (N!,N)
        All permutations of size N
    """
    def __init__(
        self, parents, kinematics_models, 
        observations, outlier_probs,
        affinity, identity):

        self.kinematics_models = kinematics_models
        self.observations = jnp.array(observations)
        self.outlier_probs = jnp.array(outlier_probs)
        self.affinity = jnp.array(affinity)
        self.identity = jnp.array(identity)
        self.parents = jnp.array(parents)
        N,T,K,C = observations.shape[:-1]
        
        # initialize params
        self.key = jax.random.PRNGKey(0)
        self.permutations = all_permutations(N)
        self.clique_affinity = (affinity[self.permutations,:][:,:,self.permutations] 
                              * jnp.eye(N).reshape(1,N,1,N,1,1,1)).sum((1,3))
        
        self.clique_affinity_row_norm = log_normalize(self.clique_affinity, axis=1)[0]
        self.clique_affinity_col_norm = log_normalize(self.clique_affinity, axis=0)[0]
        

    def initialize(self, eta=10):
        """Initialize assignments using maximum likelihood inference over 
        affinities, initial identity probabilities and keypoint proximity,
        then use these assignments to initialize the kinematics models.
        """
        self.assignments = initial_assignments(
            self.observations, self.identity,
            self.clique_affinity_row_norm,
            self.clique_affinity_col_norm,
            self.parents, eta)
        
        for model, obs, out_p in zip(self.kinematics_models,
            split_animals(self.observations,self.assignments),
            split_animals(self.outlier_probs,self.assignments)): 
            model.initialize(obs, out_p)
        

    def step(self):
        """Perform one Gibbs step. Each step consists of resampling params in 
        each kinematics model (using the current assignments), and then resampling
        assignments using log probabilities from the kinematics models. 
        """
        assign_lls = [model.obs_log_likelihood(
            self.observations, self.outlier_probs
        ) for model in self.kinematics_models]

        self.key, key = jr.split(self.key)
        self.assignments,log_likelihood = forward_backward_assignments(
            key, self.parents,
            jnp.stack(assign_lls,axis=1)+self.identity,
            self.clique_affinity_row_norm,
            self.clique_affinity_col_norm)

        for model, obs, out_p in zip(self.kinematics_models,
            split_animals(self.observations,self.assignments),
            split_animals(self.outlier_probs,self.assignments)): 
            model.step(obs, out_p)
        return log_likelihood

#-----------------------------------------------------------------------------#
#                              Gimbal body models                             #
#-----------------------------------------------------------------------------#


def fit_gimbal_model(dirs, num_states, num_iters):
    key = jr.PRNGKey(1)
    dirs = jnp.array(dirs+np.random.uniform(-1e-3,1e-3,dirs.shape))
    
    em_output = gimbal.fit.em_movMF(key, dirs[:,1:], num_states, num_iters)
    lls, E_zs, pis_em, mus_em, kappas_em = map(np.array, em_output)
    
    kappas_root = np.zeros((num_states,1))
    mus_root = np.tile(np.array([1.,0]), (num_states,1,1))
    
    kappas_em = np.concatenate([kappas_root, kappas_em], axis=1)
    mus_em = np.concatenate([mus_root, mus_em], axis=1)

    return lls, pis_em, mus_em, kappas_em



def make_gimbal2D_params(
    *, parents, node_order,
    indices_egocentric,
    obs_outlier_variance,
    obs_inlier_variance,
    pos_dt_variance,
    radii, radii_std,
    pis, mus, kappas,
    num_leapfrog_steps,
    hmc_step_size,
    **kwargs):
    
    num_joints = len(node_order)
    return {
        'crf_keypoints': jnp.array(indices_egocentric),
        'obs_inlier_location': jnp.zeros((num_joints,2)),
        'obs_outlier_location': jnp.zeros((num_joints,2)),
        'obs_outlier_variance': jnp.ones(num_joints)*obs_outlier_variance,
        'obs_inlier_variance': jnp.ones(num_joints)*obs_inlier_variance,
        'pos_radius': jnp.array(radii),
        'pos_radial_variance': jnp.array([1e8,*radii_std[1:]**2]),
        'parents': jnp.array(parents),
        'pos_dt_variance': jnp.ones(num_joints)*pos_dt_variance,
        'state_probability': jnp.array(pis),
        'state_directions': jnp.array(mus),
        'state_concentrations': jnp.array(kappas),
        'state_transition_count': jnp.ones(len(pis)),
        'obs_outlier_covariance': jnp.tile(jnp.eye(2),(num_joints,1,1))*obs_outlier_variance,
        'obs_inlier_covariance': jnp.tile(jnp.eye(2),(num_joints,1,1))*obs_inlier_variance,
        'num_leapfrog_steps': num_leapfrog_steps, 
        'step_size': hmc_step_size }



class Gimbal3D:
    def __init__(self, params, hmc_options):
        self.seed = jr.PRNGKey(0)
        params = gimbal.mcmc.initialize_parameters(params)
        self.params = self.augment_params(params)
        self.hmc_options = hmc_options
        self.samples = None
        
    def augment_params(self, params):
        params['obs_inlier_precision'] = jnp.linalg.inv(params['obs_inlier_covariance'])
        params['obs_outlier_precision'] = jnp.linalg.inv(params['obs_outlier_covariance'])
        return params
        
    def initialize(self, obs, outlier_prob):
        self.samples = []
        for i in range(obs.shape[0]):
            self.params['obs_outlier_probability'] = \
                jnp.swapaxes(outlier_prob[i], -2, -1)
            self.samples.append(gimbal.mcmc.initialize(
                self.seed, self.params, jnp.swapaxes(obs[i], -2, -3)))
                
    def step(self, obs, outlier_prob, **kwargs):
        assert self.samples is not None, 'Must run `Gimbal.initialize first'
        self.seed = jr.split(self.seed)[0]
        for i in range(obs.shape[0]):
            self.params['obs_outlier_probability'] = \
                jnp.swapaxes(outlier_prob[i], -2, -1)
            self.samples[i] = gimbal.mcmc.step(
                self.seed, self.params, jnp.swapaxes(obs[i], -2, -3), 
                self.samples[i], **self.hmc_options)
    
    def obs_log_likelihood(self, obs, outlier_prob):
        pred = jnp.stack([jax.vmap(gimbal.mcmc.project, in_axes=(0,None), out_axes=-3)(
            self.params['camera_matrices'], s['positions']) for s in self.samples])
        err = pred-jnp.moveaxis(obs, -2, -3)
        inlier_lp = gaussian_log_prob(err, self.params['obs_inlier_location'], self.params['obs_inlier_precision'])
        outlier_lp = gaussian_log_prob(err, self.params['obs_outlier_location'], self.params['obs_outlier_precision'])
        outlier_prob = jnp.swapaxes(outlier_prob, -1, -2)
        prior = jnp.stack([1-outlier_prob, outlier_prob])
        lp = jax.nn.logsumexp(jnp.stack([inlier_lp,outlier_lp]), b=prior, axis=0)
        return jnp.nan_to_num(jnp.swapaxes(lp, -1, -2))
    
    

class Gimbal2D:
    def __init__(self, params):
        self.seed = jr.PRNGKey(0)
        params = gimbal.mcmc2d.initialize_parameters(params)
        self.params = self.augment_params(params)
        self.samples = None
        
    def augment_params(self, params):
        params['obs_inlier_precision'] = jnp.linalg.inv(params['obs_inlier_covariance'])
        params['obs_outlier_precision'] = jnp.linalg.inv(params['obs_outlier_covariance'])
        return params
        
    def initialize(self, obs, outlier_prob):
        self.samples = gimbal.mcmc2d.initialize(
            self.seed, self.params, obs[...,0,:], outlier_prob[...,0])
                
    def step(self, obs, outlier_prob, **kwargs):
        assert self.samples is not None, 'Must run `Gimbal.initialize first'
        self.seed = jr.split(self.seed)[0]
        self.samples = gimbal.mcmc2d.step(
            self.seed, self.params, self.samples, 
            obs[...,0,:], outlier_prob[...,0])
    
    def obs_log_likelihood(self, obs, outlier_prob):
        err = self.samples['positions']-obs[...,0,:]
        inlier_lp = gaussian_log_prob(err, self.params['obs_inlier_location'], self.params['obs_inlier_precision'])
        outlier_lp = gaussian_log_prob(err, self.params['obs_outlier_location'], self.params['obs_outlier_precision'])
        prior = jnp.stack([1-outlier_prob, outlier_prob])[...,0]
        lp = jax.nn.logsumexp(jnp.stack([inlier_lp,outlier_lp]), b=prior, axis=0)
        return jnp.nan_to_num(lp[...,na])
    

#-----------------------------------------------------------------------------#
#                              XXX                                            #
#-----------------------------------------------------------------------------#



class DummyModel:
    def initialize(observations, outlier_probabilities): pass
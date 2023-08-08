import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import cv2
import networkx as nx
import itertools
from scipy.special import logsumexp
na = jnp.newaxis

def all_permutations(N):
    """Get all permutations of N elements as a (N!, N) array."""
    return jnp.array(list(itertools.permutations(range(N))))


def get_permutation_indexes(permutations):
    """
    Get the indexes of permutations based on their ordering
    in :py:func:`keypoint_sort.util.all_permutations`.

    Parameters
    ----------
    permutations: array of shape (n, N)
        Array of n permutations of N items.

    Returns
    -------
    indexes: array of shape (n,)
        Integer array of indexes.
    """
    reference_permutations = all_permutations(permutations.shape[-1])
    return jnp.argmax((reference_permutations[na,:,:]==permutations[:,na,:]).sum(-1),-1)


def permutation_product_indexes(N):
    """
    Generate a table of indexes for pairwise products of permutations.

    In the output array, entry [i,j] is the index of the permutation
    that results from composing permutations i and j, using the
    permutations from :py:func:`keypoint_sort.util.all_permutations`.
    """
    permutations = all_permutations(N)
    Nfac = len(permutations)
    permutation_products = permutations[:,permutations].reshape(-1,N)
    return get_permutation_indexes(permutation_products).reshape(Nfac,Nfac)


def np_io(fn): 
    """
    Converts a function involving jax arrays to one that inputs and
    outputs numpy arrays.
    """
    return lambda *args, **kwargs: jax.device_get(
        fn(*jax.device_put(args), **jax.device_put(kwargs)))


def jax_io(fn): 
    """
    Converts a function involving numpy arrays to one that inputs and
    outputs jax arrays.
    """
    return lambda *args, **kwargs: jax.device_put(
        fn(*jax.device_get(args), **jax.device_get(kwargs)))


def log_normalize(X, axis=0):
    log_norm = jax.nn.logsumexp(X, axis=axis, keepdims=True)
    return X - log_norm, log_norm


def vector_to_angle(V):
    y,x = V[...,1],V[...,0]+1e-10
    angles = (jnp.arctan(y/x)+(x>0)*jnp.pi)%(2*jnp.pi)-jnp.pi
    return angles


def angle_to_rotation_matrix(h, keypoint_dim=3):
    m = jnp.tile(jnp.eye(keypoint_dim), (*h.shape,1,1))
    m = m.at[...,0,0].set(jnp.cos(h))
    m = m.at[...,1,1].set(jnp.cos(h))
    m = m.at[...,0,1].set(-jnp.sin(h))
    m = m.at[...,1,0].set(jnp.sin(h))
    return m



def build_node_hierarchy(bodyparts, skeleton, root_node):
    """
    Define a rooted hierarchy based on the edges of a spanning tree.

    Parameters
    ----------
    bodyparts: list of str
        Ordered list of node names.

    skeleton: list of tuples
        Edges of the spanning tree as pairs of node names.

    root_node: str
        The desired root node of the hierarchy

    Returns
    -------
    node_order: array of shape (num_nodes,)
        Integer array specifying an ordering of nodes in which parents
        precede children (i.e. a topological ordering).

    parents: array of shape (num_nodes,)
        Child-parent relationships using the indexes from `node_order`, 
        such that `parent[i]==j` when `node_order[j]` is the parent of 
        `node_order[i]`.

    Raises
    ------
    ValueError
        The edges in `skeleton` do not define a spanning tree.     
    """
    G = nx.Graph()
    G.add_nodes_from(bodyparts)
    G.add_edges_from(skeleton)

    if not nx.is_tree(G):
        cycles = list(nx.cycle_basis(G))
        raise ValueError(
            'The skeleton does not define a spanning tree, '
            'as it contains the following cycles: {}'.format(cycles))
    
    if not nx.is_connected(G):
        raise ValueError(
            'The skeleton does not define a spanning tree, '
            'as it contains multiple connected components.')
    
    node_order = list(nx.dfs_preorder_nodes(G, root_node))
    parents = np.zeros(len(node_order), dtype=int)

    for i,j in skeleton:
        i,j = node_order.index(i), node_order.index(j)
        if i<j: parents[j] = i
        else: parents[i] = j

    node_order = np.array([bodyparts.index(n) for n in node_order])
    return node_order, parents
        


def interpolate_keypoints(coordinates, outliers, axis=0):
    """
    Use linear interpolation to impute the coordinates of outliers.
    
    Parameters
    ----------
    coordinates : ndarray of shape (..., dim)
        Keypoint observations.
    outliers : ndarray of shape (...)
        Binary indicator whose true entries are outlier points.
        
    Returns
    -------
    interpolated_coordinates : ndarray with same shape as `coordinates`
        Keypoint observations with outliers imputed.
    """  
    outliers = np.moveaxis(outliers, axis, 0)
    coordinates = np.moveaxis(coordinates, axis, 0)

    T, *batch_shape, dim = coordinates.shape
    outliers = outliers.reshape(T, -1)
    coordinates = coordinates.reshape(T, -1, dim)
    
    interpolated_coordinates = np.zeros_like(coordinates)
    for i in range(coordinates.shape[1]):
        xp = np.nonzero(~outliers[:,i])[0]
        for j in range(dim):
            interpolated_coordinates[:,i,j] = np.interp(
                np.arange(T), xp, coordinates[xp,i,j])

    interpolated_coordinates = interpolated_coordinates.reshape(T, *batch_shape, dim)
    interpolated_coordinates = np.moveaxis(interpolated_coordinates, 0, axis)
    return interpolated_coordinates


def transform_detections(
    coordinates, confidences, affinities, identities, 
    identity_pseudocount=1e-1, outlier_pseudocount=1e-5, 
    affinity_center=0.2, affinity_gain=20, **kwargs):
    """
    Transform confidence, affinity and identity scores into
    log probabilities that can be used for modeling, and interpolate
    missing values in the coordinates.
    
    Each variable is transformed as follows:
    - coordinates
        Linear interpolation is used to fill in missing values.
    - confidences
        Outlier probabilities are dervied as `P(outlier) = 1-confidence`
        and then clipped away from 0 and 1 by `outlier_pseudocount`
    - affinities
        Affinity scores are transformed into same-animal log probabilities
        `log(sigmoid((affinity-affinity_center)*affinity_gain))`.
    - identities
        Identity scores are transformed into log probabilities via
        `log(identity + identity_pseudocount)`.
    
    Parameters
    ----------  
    coordinates: ndarray of shape (n_individuals, n_frames, n_bodyparts, n_cameras, 2)
        Keypoint coordinates from one or more cameras.

    confidences: ndarray of shape (n_individuals, n_frames, n_bodyparts, n_cameras)
        Confidences of multi-animal keypoints detections from one or more cameras.
        
    affinities: ndarray of shape (n_individuals, n_individuals, n_frames, n_bodyparts, n_cameras)
        Part affinity field weights for each edge in the node hierarchy,
        where `affinities[i,j,t,k,c]` is the weight from keypoint `[i,t,k,c]`
        to keypoint `[j,t,parent(k),c]`.
            
    identities: ndarray of shape (n_individuals, n_individuals, n_frames, n_bodyparts, n_cameras)
        Identity weights for each keypoint, where `identities[i,j,t,k,c]`
        is the weight that keypoint `[i,t,k,c]` belongs to individual `j`.
    
    Returns
    -------
    coordinates: ndarray of shape (n_individuals, n_frames, n_bodyparts, n_cameras, 2)
        Interpolated coordinates in the same format as `coordinates`.

    outlier_probs: ndarray of shape (n_individuals, n_frames, n_bodyparts, n_cameras)
        Outlier probabilities in the same format as `confidences`.
        
    affinity_logprobs: ndarray of shape (n_individuals, n_individuals, n_frames, n_bodyparts, n_cameras)
        Normalized same-animal log probabilities.
        
    identity_logprobs: ndarray of shape (n_individuals, n_individuals, n_frames, n_bodyparts, n_cameras)
        Normalized identity log proabilities.
    """
    missing = np.isnan(coordinates).any(-1)
    coordinates = interpolate_keypoints(coordinates, missing, axis=1)
    confidences = np.where(missing, 0, confidences)

    log_sigmoid = lambda x: -np.logaddexp(0,-x)
    confidences = np.nan_to_num(confidences)
    outlier_probs = np.clip(1-confidences, outlier_pseudocount,1-outlier_pseudocount)

    affinities = np.nan_to_num(affinities)
    aff_logprobs = log_sigmoid((affinities-affinity_center)*affinity_gain)
    aff_logprobs = aff_logprobs - logsumexp(aff_logprobs, axis=1, keepdims=True)
     
    identities = np.nan_to_num(identities)
    id_logprobs = np.log(identities + identity_pseudocount)
    id_logprobs = id_logprobs - logsumexp(id_logprobs, axis=1, keepdims=True)

    return coordinates, outlier_probs, aff_logprobs, id_logprobs



def undistort_coordinates(coordinates, Ks, dist_coefs):
    """
    Wrapper for `cv2.undistortPoints` that handles NaNs, 
    multiple cameras, and batch dimensions

    Parameters
    ----------
    coordinates : array of shape (..., num_cameras, 2)
        2D coordinates of points to undistort.

    Ks : array of shape (num_cameras, 3, 3)
        Intrinsics matrix for each camera.
    
    dist_coefs : array of shape (num_cameras, 5)
        Distortion coefficients for each camera.

    Returns
    -------
    undistorted_coordinates : array of shape (..., num_cameras, 2)
        Undistorted 2D coordinates.
    """
    *batch_dims, num_cameras = coordinates.shape[:-1]
    coordinates = coordinates.reshape(-1, num_cameras, 2)
    undistorted_coordinates = np.zeros_like(coordinates)*np.nan

    for i in range(num_cameras):
        undistorted_coordinates[:,i,:] = cv2.undistortPoints(
            coordinates[na,:,i,:], Ks[i], dist_coefs[i], P=Ks[i]).squeeze(1)
        
    return undistorted_coordinates.reshape(*batch_dims, num_cameras, 2)
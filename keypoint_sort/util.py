import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import networkx as nx
from scipy.special import logsumexp
na = jnp.newaxis


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
        

def unbatch(data, labels): 
    """
    Invert :py:func:`jax_moseq.utils.batch`
 
    Parameters
    ----------
    data: ndarray, shape (num_segs, seg_length, ...)
        Stack of segmented time-series

    labels: tuples (str,int,int)
        Labels for the rows of ``data`` as tuples with the form
        (name,start,end)

    Returns
    -------
    data_dict: dict
        Dictionary mapping names to reconstructed time-series
    """     
    data_dict = {}
    keys = sorted(set([key for key,start,end in labels]))    
    for key in keys:
        length = np.max([e for k,s,e in labels if k==key])
        seq = np.zeros((int(length),*data.shape[2:]), dtype=data.dtype)
        for (k,s,e),d in zip(labels,data):
            if k==key: seq[s:e] = d[:e-s]
        data_dict[key] = seq
    return data_dict


def batch(data_dict, keys=None, seg_length=None, seg_overlap=30):
    """
    Stack time-series data of different lengths into a single array for
    batch processing, optionally breaking up the data into fixed length 
    segments. Data is 0-padded so that the stacked array isn't ragged.

    Parameters
    ----------
    data_dict: dict {str : ndarray}
        Dictionary mapping names to ndarrays, where the first dim
        represents time. All data arrays must have the same shape except
        for the first dim. 

    keys: list of str, default=None
        Optional list of names specifying which datasets to include in 
        the output and what order to put them in. Each name must be a 
        key in ``data_dict``. If ``keys=None``, names will be sorted 
        alphabetically.

    seg_length: int, default=None
        Break each time-series into segments of this length. If 
        ``seg_length=None``, the final stacked array will be as long
        as the longest time-series. 

    seg_overlap: int, default=30
        Amount of overlap between segments. For example, setting
        ``seg_length=N`` and ``seg_overlap=M`` will result in segments
        with start/end times (0, N+M), (N, 2*N+M), (2*N, 3*N+M),...

    Returns
    -------
    data: ndarray, shape (N, seg_length, ...)
        Stacked data array

    mask: ndarray, shape (N, seg_length)
        Binary indicator specifying which elements of ``data`` are not
        padding (``mask==0`` in padded locations)

    keys: list of tuples (str,int), length N
        Row labels for ``data`` consisting (name, segment_num) pairs

    """
    if keys is None: keys = sorted(data_dict.keys())
    Ns = [len(data_dict[key]) for key in keys]
    if seg_length is None: seg_length = np.max(Ns)
        
    stack,mask,labels = [],[],[]
    for key,N in zip(keys,Ns):
        for start in range(0,N,seg_length):
            arr = data_dict[key]
            end = min(start+seg_length+seg_overlap, N)
            pad_length = seg_length+seg_overlap-(end-start)
            padding = np.zeros((pad_length,*arr.shape[1:]), dtype=arr.dtype)
            mask.append(np.hstack([np.ones(end-start),np.zeros(pad_length)]))
            stack.append(np.concatenate([arr[start:end],padding],axis=0))
            labels.append((key,start,end))

    stack = np.stack(stack)
    mask = np.stack(mask)
    return stack,mask,labels


def log_normalize(X, axis=0):
    log_norm = jax.nn.logsumexp(X, axis=axis, keepdims=True)
    return X - log_norm, log_norm




def transform_detections(
    confidences, affinities, identities, 
    identity_pseudocount=1e-1, outlier_pseudocount=1e-5, 
    affinity_center=0.2, affinity_gain=20, **kwargs):
    """
    Transform confidence, affinity and identity scores into
    log probabilities that can be used for modeling. 
    
    Each variable is transformed as follows:
    - confidences
        Outlier probabilities are dervied as `P(outlier) = 1-confidence`
        and then clipped away from 0 and 1 by `outlier_pseudocount`
    - affinities
        Affinity scores are transformed into same-animal log probabilities
        `log(sigmoid((affinity-affinity_center)*affinity_gain))`.
    - identities
        Identity scores are transforme into log probabilities via
        `log(identity + identity_pseudocount)`.
    
    Parameters
    ----------  
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
    outlier_probs: ndarray of shape (n_individuals, n_frames, n_bodyparts, n_cameras)
        Outlier probabilities in the same format as `confidences`.
        
    affinity_logprobs: ndarray of shape (n_individuals, n_individuals, n_frames, n_bodyparts, n_cameras)
        Normalized same-animal log probabilities.
        
    identity_logprobs: ndarray of shape (n_individuals, n_individuals, n_frames, n_bodyparts, n_cameras)
        Normalized identity log proabilities.
    """
    log_sigmoid = lambda x: -np.logaddexp(0,-x)
    outlier_probs = np.clip(1-confidences, outlier_pseudocount,1-outlier_pseudocount)
    aff_logprobs = log_sigmoid((affinities-affinity_center)*affinity_gain)
    aff_logprobs = aff_logprobs - logsumexp(aff_logprobs, axis=1, keepdims=True)  
    id_logprobs = np.log(identities + identity_pseudocount)
    id_logprobs = id_logprobs - logsumexp(id_logprobs, axis=1, keepdims=True)
    return outlier_probs, aff_logprobs, id_logprobs


@jax.jit
def gaussian_log_prob(x, mu, sigma_inv):
    return (-((mu-x)[...,na,:]*sigma_inv*(mu-x)[...,:,na]).sum((-1,-2))/2
            +jnp.log(jnp.linalg.det(sigma_inv))/2)

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

def filter_poses(poses, parents, max_distance_std):
    radii = np.sqrt(((poses-poses[:,parents])**2).sum(-1))[:,1:]
    outliers = np.any([
        radii > np.percentile(radii,99,axis=0),
        radii < np.percentile(radii,1, axis=0)], axis=(0,2))
    radii = radii[~outliers]
    radii_mean = np.mean(radii,axis=0)
    radii_std = np.std(radii,axis=0)
    z = (radii-radii_mean)/radii_std
    poses = poses[~outliers][(np.abs(z) < max_distance_std).all(1)]
    return poses, np.pad(radii_mean,(1,0)), np.pad(radii_std,(1,0))

def standardize_poses(poses, indices_egocentric):
    poses = poses - poses.mean(1, keepdims=True)
    front = poses[:,indices_egocentric[0],:2]
    back = poses[:,indices_egocentric[1],:2]
    angle = vector_to_angle(front-back)
    rot = angle_to_rotation_matrix(angle, keypoint_dim=poses.shape[-1])
    return poses @ rot

def compute_directions(poses, parents):
    dirs = poses - poses[:, parents]
    dirs /= (np.linalg.norm(dirs, axis=-1, keepdims=True)+1e-16)
    return dirs[~np.isnan(dirs).any((1,2))]


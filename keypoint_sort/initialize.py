

from scipy.linalg import eigh
from numba import njit, prange
from numba.typed import List
import numpy as np
import jax.numpy as jnp
import tqdm
na = np.newaxis

from gimbal.util import opencv_triangulate_dlt, project
from keypoint_sort.util import all_permutations, log_normalize
from keypoint_sort.model import viterbi_assignments


@njit
def _distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

@njit
def _assign_clusters(centroids, points):
    clusters = List()
    for point in points:
        distances = np.array([_distance(point, centroid) for centroid in centroids])
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return clusters

@njit
def _compute_centroids(clusters, points, k):
    n_features = points[0].shape[0]
    centroids = np.zeros((k, n_features))
    cluster_sizes = np.zeros(k)
    for i in prange(len(clusters)):
        centroids[clusters[i]] += points[i]
        cluster_sizes[clusters[i]] += 1
    for i in prange(k):
        if cluster_sizes[i] != 0:
            centroids[i] /= cluster_sizes[i]
    return centroids

@njit
def _compute_inertia(points, centroids, clusters):
    inertia = 0.0
    for i in prange(len(points)):
        inertia += _distance(points[i], centroids[clusters[i]])**2
    return inertia


@njit
def kmeans(points, k, max_iters=100, n_init=10, tol=1e-4):
    """
    Performs K-means clustering.
    
    Parameters
    ----------
    points : numpy.ndarray, shape=(n_samples, n_features)
        The input data.
        
    k : int
        The number of clusters to form.
        
    max_iters : int, default=100
        The maximum number of iterations for each single run.
        
    n_init : int, default=10
        The number of time the k-means algorithm will be run with 
        different centroid seeds. The final results will be the best 
        output of n_init consecutive runs in terms of inertia.
        
    tol : float, optional
        Inertia tolerance for early termination.
    
    Returns
    -------
    best_clusters : list of int
        Cluster labels.
    """
    best_clusters = None
    best_inertia = np.inf
    for _ in range(n_init):
        centroids = points[np.random.choice(len(points), size=k, replace=False)]
        old_inertia = np.inf
        for _ in range(max_iters):
            clusters = _assign_clusters(centroids, points)
            centroids = _compute_centroids(clusters, points, k)
            inertia = _compute_inertia(points, centroids, clusters)
            if np.abs(old_inertia - inertia) < tol:
                break
            old_inertia = inertia
        if inertia < best_inertia:
            best_inertia = inertia
            best_clusters = clusters
    return best_clusters


def spectral_clustering(adjacency_matrix, num_clusters):
    """
    Spectral clustering of an unweighted, undirected graph.

    Parameters
    ----------
    adjacency_matrix : array, shape (n_nodes, n_nodes)
        Adjacency matrix of the graph to be clustered.

    num_clusters : int
        Number of clusters to form.
    
    Returns
    -------
    clusters : array, shape (n_nodes,)
        Cluster labels.
    """
    # calculate Laplacian, truncate to smallest eigvals
    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
    laplacian_matrix = degree_matrix - adjacency_matrix
    eigenvalues, eigenvectors = eigh(laplacian_matrix)
    X = eigenvectors[:, :num_clusters]
    
    # normalize and cluster
    X = X / np.linalg.norm(X, axis=1)[:, None]
    clusters = kmeans(X, num_clusters)
    return np.array(clusters)


@njit
def _masked_split(X, assignments):
    N,T,K,C,M = X.shape
    split_X = np.zeros_like(X)*np.nan
    for t in range(T):
        for k in range(K):
            for c in range(C):
                for i in range(N):
                    if not np.isnan(assignments[i,t,k,c]):
                        ass = int(assignments[i,t,k,c])
                        for j in range(M):
                            split_X[ass,t,k,c,j] = X[i,t,k,c,j]
    return split_X
    
    
def masked_split_animals(X, assignments):
    """
    Use assignments to sort an array X of shape (N,T,K,C,[dim]) such 
    that all keypoints for animal i are in slice [i,:,:,:,[dim]]

    This function is similar to :py:func:`keypoint_sort.model.split_animals`
    but operates on numpy arrays rather than jax arrays and can handle NaNs
    in the assignments.

    Parameters
    ----------
    X : jnp.ndarray, shape (N,T,K,C,[dim])
        Array to be sorted by assignments.
        
    assignments : jnp.ndarray, shape (N,T,K,C)
        Identity assignments where assignments[i,t,k,c] = j means that 
        the i'th instance of keypoint k in frame t and camera c belongs 
        to animal j.
    """
    shape = X.shape
    X = X.reshape(*shape[:4],-1)
    X_split = _masked_split(X, assignments)
    return X_split.reshape(shape)


def triangulation_errors(coordinates, projection_matrices, verbose=True):
    """
    Calculate triangulation errors for each pair of keypoints across cameras.
    
    Parameters
    ----------
    coordinates: ndarray of shape (n_indivs, n_frames, n_bodyparts, n_cams, 2)
        Keypoint coordinates from one or more cameras.

    projection_matrices : array, shape (n_cams, 3, 4)
        Projection matrices (intrinsic @ [R | t]) for each camera

    verbose : bool, default=True
        Whether to display a progress bar.

    Returns
    -------
    errors : ndarray, shape (n_indivs, n_indivs, n_frames, n_bodyparts, n_cams, n_cams)
        Triangulation errors for each pair of keypoints across cameras,
        where `errors[i1,i1,t,k,c1,c2]` is the triangulation error 
        between instance i1 in camera c1 and instance i2 in camera c2.
        (Only the upper triangle of the last two dimensions is filled.)
    """
    N,T,K,C = coordinates.shape[:4]
    errors = np.zeros((N,N,T,K,C,C))
    
    camera_pairs = [(i,j) for i in range(1,C) for j in range(i)]
    for i,j in tqdm.tqdm(camera_pairs, disable=not verbose, unit='camera pair'):

        ys = np.stack([
            np.repeat(coordinates[:,na,:,:,i],N,axis=1),
            np.repeat(coordinates[na,:,:,:,j],N,axis=0)])
        
        mask = ~np.isnan(ys).any((0,-1))
        ys = ys.reshape(2,-1,2)[:,mask.flatten(),:]
        ys_triangulated = np.array(opencv_triangulate_dlt(projection_matrices[[i,j]], ys))
        
        ys_projected = np.stack([
            project(projection_matrices[i], ys_triangulated), 
            project(projection_matrices[j], ys_triangulated)])
        
        errors[:,:,:,:,i,j][mask] = np.linalg.norm(ys - ys_projected, axis=-1).mean(0)
    
    mask = ~np.isnan(coordinates).any(-1)
    errors *= mask[:,na,:,:,:,na] * mask[na,:,:,:,na,:]
    errors = np.where(errors==0, np.nan, errors)
    return errors


def jenson_shannon(P1, P2, axis=-1, pseudocount=1e-6):
    """
    Jenson-Shannon divergence between two probability distributions.

    Parameters
    ----------
    P1,P2 : ndarrays
        Possibly unnormalized probability distributions to compare.

    axis : int, default=-1
        Axis along which `P1` and `P2` define probability distributions.

    pseudocount : float, default=1e-6
        Pseudocount to add to each probability distribution before
        normalizing.

    Returns
    -------
    divergence : ndarray
        Jenson-Shannon divergence between `P1` and `P2`. The shape is
        the same as `P1` and `P2` with the `axis` dimension removed.
    """
    def _norm(P):
        P = P + pseudocount
        return P / P.sum(axis, keepdims=True)

    def _kl(P1, P2):
        return (P1 * np.log(P1/P2)).sum(axis)
        
    P1 = _norm(P1)
    P2 = _norm(P2)
    M = (P1 + P2)/2
    return _kl(P1,M)/2 + _kl(P2,M)/2


def identity_divergences(identities, verbose=True):
    """
    Calculate the Jenson-Shannon divergence between identity probabilities
    for each pair of keypoints across cameras.
    
    Parameters
    ----------
    identities : ndarray, shape (n_indivs, n_indivs, n_frames, n_bodyparts, n_cams)
        Identity probabilities for each keypoint, where `identities[i,j,t,k,c]`
        is the probability that keypoint `[i,t,k,c]` belongs to individual `j`.

    verbose : bool, default=True
        Whether to display a progress bar.

    Returns
    -------
    divergences : ndarray, shape (n_indivs, n_indivs, n_frames, n_bodyparts, n_cams, n_cams)
        Jenson-Shannon divergence between identity probabilities for each
        pair of keypoints across cameras, where `divergences[i1,i1,t,k,c1,c2]`
        is the identity divergence between instance i1 in camera c1 and instance
        i2 in camera c2. (Only the upper triangle of the last two dimensions
        is filled.)
    """
    N,_,T,K,C = identities.shape
    divergences = np.zeros((N,N,T,K,C,C))
    camera_pairs = [(i,j) for i in range(1,C) for j in range(i)]
    for i,j in tqdm.tqdm(camera_pairs, unit='camera pair', disable=not verbose):
        P1 = np.exp(np.repeat(identities[:,na,:,:,:,i],N,axis=1))
        P2 = np.exp(np.repeat(identities[na,:,:,:,:,j],N,axis=0))
        divergences[:,:,:,:,i,j] = jenson_shannon(P1, P2, axis=2)
    return divergences


@njit
def _dfs(v, visited, adj_matrix):
    visited[v] = True
    results = [v]
    for neighbor in np.where(adj_matrix[v] > 0)[0]:  # Find all neighbors
        if visited[neighbor] == 0:
            results.extend(_dfs(neighbor, visited, adj_matrix))
    return results

@njit
def connected_components(adj_matrix):
    """Get the connected components of a graph."""
    n = adj_matrix.shape[0]  # Number of nodes
    visited = np.zeros(n, dtype=np.bool_)
    components = []
    for v in range(n):
        if visited[v] == 0:
            component = _dfs(v, visited, adj_matrix)
            components.append(component)
    return components

@njit
def filtered_components(adj_matrix, scores, N):
    """
    Find connected components of a graph and return only the top N
    by total node score.

    Parameters
    ----------
    adj_matrix : ndarray, shape (n_nodes, n_nodes)
        Adjacency matrix of the graph.

    scores : ndarray, shape (n_nodes,)
        Scores for each node.

    N : int
        Maximum number of components to return.

    Returns
    -------
    components : list of lists
        List of connected components, where each component is a list
        of node indices.
    """
    components = connected_components(adj_matrix)
    components = [c for c in components if len(c) > 1]

    # If there are more than N components, keep only the top N by total node score
    if len(components) > N:
        total_scores = np.zeros(len(components))
        for i, component in enumerate(components):
            total_score = 0
            for node in component:
                total_score += scores[node]
            total_scores[i] = total_score
        top_N_indices = np.argsort(total_scores)[-N:]  # Use argsort instead of argpartition
        components = [components[i] for i in top_N_indices]
    return components


def group_across_cameras(errors, divergences, outlier_probs, error_threshold=15, 
                         div_threshold=0.1, verbose=True):
    """
    Group keypoint observations across cameras based on pairwise
    triangulation errors and identity divergences.

    A graph is constructed where edges represent pairs of keypoints
    with sufficiently low triangulation error and identity divergence.
    Connected components of this graph are then used to group keypoints.
    If the connected components yield an invalid grouping (i.e. a grouping
    where a single camera contains multiple instances of the same individual),
    then the connected components are clustered using spectral clustering.
    
    Parameters
    ----------
    errors : ndarray, shape (n_indivs, n_indivs, n_frames, n_bodyparts, n_cams, n_cams)
        Triangulation errors for each pair of keypoints across cameras,
        where `errors[i1,i1,t,k,c1,c2]` is the triangulation error 
        between instance i1 in camera c1 and instance i2 in camera c2.

    divergences : ndarray, shape (n_indivs, n_indivs, n_frames, n_bodyparts, n_cams, n_cams)
        Jenson-Shannon divergence between identity probabilities for each 
        pair of keypoints across cameras, where `divergences[i1,i1,t,k,c1,c2]` 
        is the identity divergence between instance i1 in camera c1 and instance 
        i2 in camera c2.

    outlier_probs : ndarray, shape (n_indivs, n_frames, n_bodyparts, n_cams)
        Outlier probabilities for each keypoint, used to prioritize
        which keypoints get included in the groupings.

    error_threshold : float, default=15
        Maximum triangulation error for two keypoints to be connected by an edge.

    div_threshold : float, default=0.1
        Maximum identity divergence for two keypoints to be connected by an edge.

    verbose : bool, default=True
        Whether to display a progress bar.

    Returns
    -------
    groupings : ndarray, shape (n_indivs, n_frames, n_bodyparts, n_cams)
        Groupings of keypoints into individuals. Each entry is the
        index of the group to which the keypoint belongs, or NaN
        if the keypoint does not belong to a group.
    """
    N,T,K,C = outlier_probs.shape
    As = np.all([errors<error_threshold, divergences<div_threshold], axis=0)
    As = np.transpose(As, axes=(2,3,4,0,5,1)).reshape(T,K,C*N,C*N)
    node_scores = np.moveaxis(1-outlier_probs,0,3).reshape(T,K,C*N)

    groupings = np.zeros((N,T,K,C))*np.nan
    camera_index = np.repeat(np.arange(C),N)
    instance_index = np.tile(np.arange(N),C)

    for t in tqdm.trange(T, disable=not verbose, unit='frames'):
        for k in range(K):
            A = As[t,k] + As[t,k].T
            ix = A.any(1).nonzero()[0]

            # first try connected components
            comps = filtered_components(A[ix,:][:,ix], node_scores[t,k][ix], N) 
            if len(comps)==1 and np.bincount(camera_index[ix[comps[0]]]).max()>1:
                # now try clustering
                clus = spectral_clustering(A[ix,:][:,ix], N)
                comps = [(clus==i).nonzero()[0] for i in range(N) if (clus==i).sum()>1]

            for i,comp in enumerate(comps):
                if np.bincount(camera_index[ix[comp]]).max()==1:
                    groupings[:,t,k,:][(instance_index[ix[comp]],camera_index[ix[comp]])] = i
    return groupings



@njit
def _get_group_affinities(groupings, affinities, parents):
    N,T,K,C = groupings.shape
    group_affinities = np.zeros((N,N,T,K))
    for t in range(T):
        for child in range(K):
            parent = parents[child]
            if child != parent:
                for c in range(C):
                    for i in range(N):
                        for j in range(N):
                            child_grp = groupings[i,t,child,c]
                            parent_grp = groupings[j,t,parent,c]
                            if not (np.isnan(child_grp) or np.isnan(parent_grp)):
                                aff = affinities[i,j,t,child,c]
                                group_affinities[int(child_grp),int(parent_grp),t,child] += aff
    return group_affinities


@njit
def _get_group_identities(groupings, identities):
    N,T,K,C = groupings.shape
    group_identities = np.zeros((N,N,T,K))
    for t in range(T):
        for k in range(K):
            for c in range(C):
                for i in range(N):
                    grp = groupings[i,t,k,c]
                    if not np.isnan(grp):
                        for j in range(N):
                            group_identities[int(grp),j,t,k] += identities[i,j,t,k,c]
    return group_identities


@njit
def _apply_assignments(groupings, assignments):
    N,T,K,C = groupings.shape
    mapped_assignments = np.zeros((N,T,K,C))*np.nan
    for t in range(T):
        for k in range(K):
            for c in range(C):
                for i in range(N):
                    if not np.isnan(groupings[i,t,k,c]):
                        grp = int(groupings[i,t,k,c])
                        mapped_assignments[i,t,k,c] = assignments[grp,t,k]
    return mapped_assignments
                    

def assemble_groups(groupings, affinities, identities, parents):
    """
    Assemble multi-camera groups of keypoints into individual animals.

    Parameters
    ----------
    groupings : ndarray, shape (n_indivs, n_frames, n_bodyparts, n_cams)
        Groupings of keypoints into individuals. Each entry is the
        index of the group to which the keypoint belongs, or NaN
        if the keypoint does not belong to a group.

    affinities: ndarray of shape (n_indivs, n_indivs, n_frames, n_bodyparts, n_cams)
        Part affinity field weights for each edge in the node hierarchy,
        where `affinities[i,j,t,k,c]` is the weight from keypoint `[i,t,k,c]`
        to keypoint `[j,t,parent(k),c]`.
            
    identities: ndarray of shape (n_indivs, n_indivs, n_frames, n_bodyparts, n_cams)
        Identity weights for each keypoint, where `identities[i,j,t,k,c]`
        is the weight that keypoint `[i,t,k,c]` belongs to individual `j`.

    parents : ndarray, shape (n_bodyparts,)
        Array of parent indices for each keypoint where parents[i] = j
        means that keypoint i is a child of keypoint j.

    Returns
    -------
    assignments : ndarray, shape (n_indivs, n_frames, n_bodyparts, n_cams)
        Assignments of keypoints to individuals, with unassigned keypoints
        set to NaN.
    """
    N,T,K,C = groupings.shape
    group_affinities = jnp.array(_get_group_affinities(groupings, affinities, parents))[...,na]
    group_identities = jnp.array(_get_group_identities(groupings, identities))[...,na]

    parents = jnp.array(parents)
    permutations = jnp.array(all_permutations(groupings.shape[0]))
    
    clique_affinity = (group_affinities[permutations,:][:,:,permutations]*jnp.eye(N).reshape(1,N,1,N,1,1,1)).sum((1,3))
    clique_affinity_row_norm = log_normalize(clique_affinity, axis=1)[0]
    clique_affinity_col_norm = log_normalize(clique_affinity, axis=0)[0]

    assignments_init = np.array(viterbi_assignments(
            parents, group_identities,
            clique_affinity_row_norm,
            clique_affinity_col_norm).squeeze())

    assignments = _apply_assignments(groupings, assignments_init)
    return assignments


def initial_assignments(coordinates, outlier_probs, affinities, identities, 
                        projection_matrices, parents, outlier_threshold=0.2,
                        div_threshold=0.1, error_threshold=15, verbose=True):
    
    coordinates = np.where(outlier_probs[...,na]<.2, coordinates, np.nan)

    if verbose: print('calculating pairwise triangulation errors')
    errors = triangulation_errors(coordinates, projection_matrices, verbose=verbose)

    if verbose: print('calculating JS divergences between identity probabilities')
    divergences = identity_divergences(identities, verbose=verbose)

    if verbose: print('grouping keypoint observations across cameras')
    groupings = group_across_cameras(errors, divergences, outlier_probs, 
                                     error_threshold=error_threshold, 
                                     div_threshold=div_threshold, verbose=verbose)

    if verbose: print('assembling multi-camera groups into individuals')
    assignments = assemble_groups(groupings, affinities, identities, parents)

    return assignments



def sample_assemblies(coordinates, assignments):
    coords = masked_split_animals(coordinates, assignments)
    return coords


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


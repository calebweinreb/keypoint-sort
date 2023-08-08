import os
import yaml
import tqdm
import h5py
import pickle
import numpy as np
import multiprocessing

import warnings
from textwrap import fill
warnings.formatwarning = lambda msg, *a: str(msg)

from .util import build_node_hierarchy

from gimbal.util_io import load_camera_parameters


def _build_yaml(sections, comments):
    text_blocks = []
    for title,data in sections:
        centered_title = f' {title} '.center(50, '=')
        text_blocks.append(f"\n\n{'#'}{centered_title}{'#'}")
        for key,value in data.items():
            text = yaml.dump({key:value}).strip('\n')
            if key in comments: text = f"\n{'#'} {comments[key]}\n{text}"
            text_blocks.append(text)
    return '\n'.join(text_blocks)
        

def generate_config(project_dir, **kwargs):
    """
    Generate a `config.yml` file with project settings. Default 
    settings will be used unless overriden by a keyword argument.
    
    Parameters
    ----------
    project_dir: str 
        A file `config.yml` will be generated in this directory.
    
    kwargs
        Custom project settings.  
    """
    
    def _update_dict(new, original):
        return {k:new[k] if k in new else v for k,v in original.items()} 
    

    gimbal = _update_dict(kwargs, {
        'num_sample_poses': 500,
        'num_sample_session': 10,
        'num_iters': 500,
        'num_states': 50,
        'num_animals': 2,
        'obs_outlier_variance': 1e6,
        'obs_inlier_variance': 10,
        'pos_dt_variance': 5,
        'num_leapfrog_steps': 5,
        'hmc_step_size': .5})

    anatomy = _update_dict(kwargs, {
        'bodyparts': ['BODYPART1','BODYPART2','BODYPART3'],
        'skeleton': [['BODYPART1','BODYPART2'], ['BODYPART2','BODYPART3']],
        'root_node': ['BODYPART1'],
        'root_edge': ['BODYPART1','BODYPART2']})
        
    other = _update_dict(kwargs, {
        'dataset_info': {},
        'individuals': None,
        'video_dir': '',
        'keypoint_colormap': 'autumn',
        'seg_length': 10000 })
       
    
    comments = {
        'individuals': 'ordered list of names of individuals; used for maDLC projects where `identity=True`',
        'keypoint_colormap': 'colormap used for visualization; see `matplotlib.cm.get_cmap` for options',
        'video_dir': 'directory with videos from which keypoints were derived',
        'session_name_suffix': 'suffix used to match videos to session names; this can usually be left empty (see `find_matching_videos` for details)',
        'bodyparts': 'used to access columns in the keypoint data',
        'skeleton': 'hierarchy of bodyparts used by GIMBAL; must be a tree (no cycles)',
        'root_node': 'node at the root of the skeleton',
        'root_edge': 'edge used to define the default heading angle',
        'seg_length': 'data are broken up into segments to parallelize fitting',
        'max_distance_std': 'maximum standard deviation of edge lengths; used to filter poses for fitting GIMBAL parameters',
        'min_confidence': 'minimum confidence of keypoints; used to filter poses for fitting GIMBAL parameters',
        'num_sample_poses': 'number of poses to sample when fitting GIMBAL parameters',
        'num_sample_sessions': 'max number of recording sessions to sample when fitting GIMBAL parameters',
        'num_iters': 'number of EM iterationswhen fitting GIMBAL parameters',
        'num_states': 'number of states in the GIMBAL model'}

    sections = [
        ('ANATOMY', anatomy),
        ('GIMBAL', gimbal),
        ('OTHER', other)
    ]

    with open(os.path.join(project_dir,'config.yml'),'w') as f: 
        f.write(_build_yaml(sections, comments))
                          
        
def check_config_validity(config):
    """
    Check if the config is valid.

    To be valid, the config must satisfy the following criteria:
        - For each pair in `config["skeleton"]`, both elements 
          also in `config["bodyparts"]` 
        - The edges in `config["skeleton"]` form a spanning tree
        - `config["root_node"]` is in `config["bodyparts"]` 
        - `config["root_edge"]` is in `config["skeleton"]` 

    Parameters
    ----------
    config: dict 

    Returns
    -------
    validity: bool
    """
    error_messages = []
    
    # check anatomy            
    for bodypart in sum(config['skeleton'],[]):
        if not bodypart in config['bodyparts']:
            error_messages.append(
                f'ACTION REQUIRED: `skeleton` contains {bodypart} '
                'which is not one of the options in `bodyparts`.')
            
    if not config['root_node'] in config['bodyparts']:
        error_messages.append(
            f'ACTION REQUIRED: `root_node` "{config["root_node"]}" '
            'is not one of the options in `bodyparts`.')
     
    if ((not config['root_edge'] in config['skeleton']) and
        (not config['root_edge'][::-1] in config['skeleton'])): 
        error_messages.append(     
            f'ACTION REQUIRED: `root_edge` "{config["root_edge"]}" '
            'is not one of the options in `skeleton`.')
        
    # check skeleton is a tree
    try: 
        build_node_hierarchy(config['bodyparts'], config['skeleton'], config['root_node'])
    except ValueError as e: 
        error_messages.append(
            'ACTION REQUIRED: `skeleton` does not form a spanning tree. '
            'The following error was raised when trying to build a tree: '
            f'{e}')

    if len(error_messages)==0: 
        return True
    for msg in error_messages: 
        print(fill(msg, width=70, subsequent_indent='  '), end='\n\n')
    return False
            

def load_config(project_dir, check_if_valid=True):
    """
    Load a project config file.
    
    Parameters
    ----------
    project_dir: str
        Directory containing the config file
        
    check_if_valid: bool, default=True
        Check if the config is valid using 
        :py:func:`keypoint_sort.io.check_config_validity`
        
    build_indexes: bool, default=True
        Add the following items to the config:

        - node_order: jax array
            Topological ordering of nodes (root first)
        - parents: jax array
            The parent of each node, using indexes derived from `node_order`
        - indices_egocentric
            The root edge as (parent,child), using indexes derived from `node_order`

    Returns
    -------
    config: dict
    """
    config_path = os.path.join(project_dir,'config.yml')
    
    with open(config_path, 'r') as stream:  
        config = yaml.safe_load(stream)

    if check_if_valid: 
        check_config_validity(config)
        
    node_order, parents = build_node_hierarchy(
        config['bodyparts'], config['skeleton'], config['root_node'])
    
    root_edge = (config['bodyparts'].index(config['root_edge'][0]),
                 config['bodyparts'].index(config['root_edge'][1]))
    indices_egocentric = np.in1d(node_order, root_edge).nonzero()[0]
    
    config['parents'] = parents
    config['node_order'] = node_order
    config['indices_egocentric'] = indices_egocentric
        
    return config


def update_config(project_dir, **kwargs):
    """
    Update the config file stored at `project_dir/config.yml`.
     
    Use keyword arguments to update key/value pairs in the config.

    Examples
    --------
    To update `video_dir` to `/path/to/videos`::

      >>> update_config(project_dir, video_dir='/path/to/videos')
      >>> print(load_config(project_dir)['video_dir'])
      /path/to/videos
    """
    config = load_config(project_dir, check_if_valid=False)
    config.update(kwargs)
    generate_config(project_dir, **config)
    
        
def setup_project(project_dir, overwrite=False, deeplabcut_config=None, **options):
    """
    Setup a project directory with the following structure::

        project_dir
        └── config.yml
    
    Parameters
    ----------
    project_dir: str 
        Path to the project directory (relative or absolute)
        
    deeplabcut_config: str, default=None
        Path to a deeplabcut config file. Relevant settings, including
        `'bodyparts'`, `'skeleton'`, and `'video_dir'` will be 
        imported from the deeplabcut config and used to initialize the 
        config. (overrided by kwargs). 

    overwrite: bool, default=False
        Overwrite any config.yml that already exists at the path
        `{project_dir}/config.yml`

    options
        Used to initialize config file. Overrides default settings
    """
    if os.path.exists(project_dir) and not overwrite:
        print(fill(
            f'The directory `{project_dir}` already exists. Use '
            '`overwrite=True` or pick a different name'))
        return
    
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)

    if deeplabcut_config is not None: 
        dlc_options = {}
        with open(deeplabcut_config, 'r') as stream:           
            dlc_config = yaml.safe_load(stream)
            
            if dlc_config is None:
                raise RuntimeError(
                    f'{deeplabcut_config} does not exists or is not a'
                    ' valid yaml file')
                
            assert 'multianimalproject' in dlc_config and dlc_config['multianimalproject'], fill(
                'Config initialization is only supported for multianimal-DLC')
            
            assert 'identity' in dlc_config and dlc_config['identity'], fill(
                'multianimal deeplabcut integration requires running deeplabcut with '
                '`identity=True` (see deeplabcut docs [LINK])')
                
            dlc_options['video_dir'] = os.path.join(dlc_config['project_path'],'videos')
            dlc_options['individuals'] = dlc_config['individuals']
            dlc_options['bodyparts'] = dlc_config['multianimalbodyparts']
            dlc_options['skeleton'] = dlc_config['skeleton']
            
        options = {**dlc_options, **options}

    generate_config(project_dir, **options)
     

def load_calibration(path):
    """
    Load a camera calibration file.

    The file should be in hdf5 format and contain a group called
    `camera_parameters` and include the following datasets:

        - `camera_names`: list of camera names
        - `rotation`: rotation matrices (n_cameras, 3, 3)
        - `translation`: translation vectors (n_cameras, 3)
        - `intrinsic`: intrinsic camera matrices (n_cameras, 3, 3)
        - `dist_coefs`: distortion coefficients (n_cameras, 5)

    Parameters
    ----------
    path: str
        Path to the calibration file

    Returns
    -------
    camera_names: list
        Ordered list of camera names
    
    projection_matrices: array, shape (n_cameras, 3, 4)
        Projection matrices (intrinsic @ [R | t]) for each camera

    rotation: array, shape (n_cameras, 3, 3)
        Rotation matrices from the world coordinate system to each
         camera's coordinate system.

    translation: array, shape (n_cameras, 3)
        Translation vectors from the world coordinate system to each
        camera's coordinate system.

    intrinsic: array, shape (n_cameras, 3, 3)
        Intrinsic camera matrices

    dist_coefs: array, shape (n_cameras, 5)
        Distortion coefficients for in OpenCV format (k1, k2, p1, p2, k3)
    """
    with h5py.File(path, 'r') as f:
        rotation = f['camera_parameters']['rotation'][()]
        translation = f['camera_parameters']['translation'][()]
        intrinsic = f['camera_parameters']['intrinsic'][()]
        dist_coefs = f['camera_parameters']['dist_coefs'][()]
        camera_names = f['camera_parameters']['camera_names'][()]
        camera_names = [name.decode('utf-8') for name in camera_names]

    extrinsics = np.concatenate((rotation, translation[:,:,np.newaxis]), axis=2)
    projection_matrices = np.matmul(intrinsic, extrinsics)
    return camera_names, projection_matrices, rotation, translation, intrinsic, dist_coefs


def load_matched_frames(path):
    """
    Load a table of matched frames for a set of videos.

    For multi-camera recordings that have distinct start/end times
    or dropped frames, this table can be used for synchronization.
    The header should be a list of camera names and each row should
    contain a set of frame indexes that are temporally aligned across
    each video. For example, the table below indicates that frame 0
    of camera1 corresponds to frame 1 of camera2::

        camera1,camera2
        0,1
        ...

    Parameters
    ----------
    path: str
        Path to the matched frames table

    Returns
    -------
    matched_frames: dict
        Dictionary mapping camera names to arrays of frame indexes
    """
    camera_names = open(path, 'r').readline().strip('\n').split(',')
    frame_indexes = np.loadtxt(path, delimiter=',', skiprows=1, dtype=int)
    matched_frames = {name: ixs for name, ixs in zip(camera_names, frame_indexes.T)}
    return matched_frames


def save_detections_to_h5(filepath, coordinates, confidences, affinities, identities):
    """
    Save keypoint detections to an HDF5 file.

    The file will contain the the keys `coordinates`, `confidences`,
    `affinities`, and `identities` which are described below.

    Parameters
    ----------
    coordinates: ndarray of shape (n_individuals, n_frames, n_bodyparts, 2)
        Coordinates of multi-animal keypoints detections.
        
    confidences: ndarray of shape (n_individuals, n_frames, n_bodyparts)
        Confidences of multi-animal keypoints detections.
        
    affinities: ndarray of shape (n_individuals, n_individuals, n_frames, n_bodyparts)
        Part affinity field weights for each edge in the node hierarchy,
        where `affinities[i,j,t,k]` is the weight from keypoint `[i,t,k]`
        to keypoint `[j,t,parent(k)]`.
            
    identities: ndarray of shape (n_individuals, n_individuals, n_frames, n_bodyparts)
        Identity weights for each keypoint, where `identities[i,j,t,k]`
        is the weight that keypoint `[i,t,k]` belongs to individual `j`.
    """
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('coordinates', data=coordinates)
        f.create_dataset('confidences', data=confidences)
        f.create_dataset('affinities', data=affinities)
        f.create_dataset('identities', data=identities)


def load_detections_from_h5(filepath):
    """
    Load keypoint detections from an HDF5 file.

    The file should contain the the keys `coordinates`, `confidences`,
    `affinities`, and `identities` that point to arrays with the format
    described below.

    Parameters
    ----------
    filepath: str
        Path to an HDF5 file containing keypoint detections.

    Returns
    -------
    coordinates: ndarray of shape (n_individuals, n_frames, n_bodyparts, 2)
        Coordinates of multi-animal keypoints detections.
        
    confidences: ndarray of shape (n_individuals, n_frames, n_bodyparts)
        Confidences of multi-animal keypoints detections.
        
    affinities: ndarray of shape (n_individuals, n_individuals, n_frames, n_bodyparts)
        Part affinity field weights for each edge in the node hierarchy,
        where `affinities[i,j,t,k]` is the weight from keypoint `[i,t,k]`
        to keypoint `[j,t,parent(k)]`.
            
    identities: ndarray of shape (n_individuals, n_individuals, n_frames, n_bodyparts)
        Identity weights for each keypoint, where `identities[i,j,t,k]`
        is the weight that keypoint `[i,t,k]` belongs to individual `j`.
    """
    with h5py.File(filepath, 'r') as f:
        coordinates = f['coordinates'][:]
        confidences = f['confidences'][:]
        affinities = f['affinities'][:]
        identities = f['identities'][:]
    return coordinates, confidences, affinities, identities


def load_multicamera_detections(keypoint_detections, calibration_path, 
                                matched_frames_path=None, **kwargs):
    """
    Load and merge keypoints detections from multiple cameras.

    The detections for each camera are stacked in the order defined
    by the calibration file. If `matched_frames_path` is provided, 
    the detections will be synchronized using the matched frames table.

    Parameters
    ----------
    keypoint_detections : dict
        Dictionary mapping camera names to h5 files with keypoint detections.
        If no extension is provided, the extension ".h5" is assumed.
        The files should have the format expected by
        :py:func:`keypoint_sort.io.load_detections_from_h5`.

    calibration_path : str
        Path to the camera calibration file.

    matched_frames_path : str, default=None
        Path to the matched frames file. If None, frames from each video
        are assumed to be already synchronized. See 
        :py:func:`keypoint_sort.io.load_matched_frames` for more details
        about the format of the matched frames file.

    Returns
    -------
    coordinates : ndarray of shape (n_individuals, n_frames, n_bodyparts, n_cameras, 2)
        Coordinates of multi-animal keypoints detections for each camera.

    confidences : ndarray of shape (n_individuals, n_frames, n_bodyparts, n_cameras)
        Confidences of multi-animal keypoints detections for each camera.

    affinities : ndarray of shape (n_individuals, n_individuals, n_frames, n_bodyparts, n_cameras)
        Part affinity field weights for each edge in the node hierarchy,
        where `affinities[i,j,t,k,c]` is the weight from keypoint `[i,t,k]`
        to keypoint `[j,t,parent(k)]` in camera `c`.

    identities : ndarray of shape (n_individuals, n_individuals, n_frames, n_bodyparts, n_cameras)
        Identity weights for each keypoint, where `identities[i,j,t,k,c]`
        is the weight that keypoint `[i,t,k]` belongs to individual `j` in camera `c`.
    """
    # Load calibration
    camera_names = load_calibration(calibration_path)[0]

    # Load detections
    detections = {}
    for camera_name,filepath in keypoint_detections.items():
        if not (filepath.endswith('.h5') or filepath.endswith('.hdf5')):
            filepath += '.h5'
        detections[camera_name] = load_detections_from_h5(filepath)

    # Load matched frames
    if matched_frames_path is not None:
        matched_frames = load_matched_frames(matched_frames_path)
    else:
        num_frames = {camera_name:d[0].shape[1] for camera_name,d in detections.items()}
        assert len(set(num_frames.values())) == 1, \
            'All detections must have the same number of frames if `matched_frames_path` is not provided.'
        matched_frames = {camera_name:np.arange(num_frames[camera_name]) for camera_name in camera_names}

    # Merge detections
    coordinates, confidences, affinities, identities = [], [], [], []

    for camera_name in camera_names:
        ixs = matched_frames[camera_name]
        coordinates.append(detections[camera_name][0][:, ixs])
        confidences.append(detections[camera_name][1][:, ixs])
        affinities.append(detections[camera_name][2][:, :, ixs])
        identities.append(detections[camera_name][3][:, :, ixs])
    
    coordinates = np.stack(coordinates, axis=-2)
    confidences = np.stack(confidences, axis=-1)
    affinities = np.stack(affinities, axis=-1)
    identities = np.stack(identities, axis=-1)

    return coordinates, confidences, affinities, identities



def load_deeplabcut_full(filepath, bodyparts, node_order, parents, 
                         individuals=None, recorded_individuals=None, 
                         verbose=True, **kwargs):
    """
    Load multi-animal DeepLabCut results from a "_full.p" file.

    - Bodyparts are reordered according to `node_order`.
    - The PAF graph is restructured according to `parents`.
    - The number of individuals is set to `len(recorded_individuals)`
      if it is provided, otherwise to `len(individuals)`. All data
      is truncated and/or NaN padded to match this number.

    Parameters
    ----------
    filepath : str
        Path to the DeepLabCut results. 

    bodyparts : list of str
        List of bodypart names.

    node_order : list of int
        Node order to be used during modeling. For example, if 
        `node_order[0]=5` then after reordering the first node
        will be `bodyparts[5]`.

    parents : list of int
        Child-parent relationships using the indexes from `node_order`, 
        such that `parent[i]==j` when `node_order[j]` is the parent of 
        `node_order[i]`.

    individuals : list of str
        Ordered of list of individuals used for training DeepLabCut.

    recorded_individuals : list of str, default=None
        Subset of individuals in the current recording. Will only be
        used if `individuals` is specified as well.
    
    verbose : bool, default=True
        If True, print status updates and show progress bars.

    Returns
    -------
    coordinates: ndarray of shape (n_individuals, n_frames, n_bodyparts, 2)
        Coordinates of multi-animal keypoints detections.
        
    confidences: ndarray of shape (n_individuals, n_frames, n_bodyparts)
        Confidences of multi-animal keypoints detections.
        
    affinities: ndarray of shape (n_individuals, n_individuals, n_frames, n_bodyparts)
        Part affinity field weights for each edge in the node hierarchy,
        where `affinities[i,j,t,k]` is the weight from keypoint `[i,t,k]`
        to keypoint `[j,t,parent(k)]`.
            
    identities: ndarray of shape (n_individuals, n_individuals, n_frames, n_bodyparts)
        Identity weights for each keypoint, where `identities[i,j,t,k]`
        is the weight that keypoint `[i,t,k]` belongs to individual `j`.
    """
    # initialize `recorded_individuals`
    if recorded_individuals is None: 
        recorded_individuals = individuals
        id_channels = np.arange(len(individuals))
    else:
        id_channels = np.array([individuals.index(i) for i in recorded_individuals])

    # load data from each file
    if verbose: print(f'Loading {filepath}')
    results = pickle.load(open(filepath, 'rb'))
    metadata = results['metadata']

    # get dimensions
    T = metadata['nframes']
    K = len(bodyparts)            # number of keypoints
    N = len(recorded_individuals) # number of individuals 

    coordinates = np.zeros((N,T,K,2))*np.nan
    confidences = np.zeros((N,T,K))
    affinities = np.zeros((N,N,T,K))
    identities = np.zeros((N,N,T,K))

    # make sure bodypart names match
    assert tuple(metadata['all_joints_names'])==tuple(bodyparts), fill(
        f'`bodyparts` does not match `all_joints_names` in the metadata of {filepath}')
    
    # make sure PAF graph contains required edges
    paf_graph = set(map(tuple,np.sort(np.argsort(node_order)[np.array(metadata['PAFgraph'])],axis=1)))
    required_pafs = set(map(tuple,np.array([parents,np.arange(len(parents))]).T[1:]))
    missing_pafs = [[bodyparts[node_order[i]] for i in paf] for paf in required_pafs - paf_graph]
    assert required_pafs.issubset(paf_graph), fill(
        f'PAF graph in {filepath} does not match `parents`. The following edges are missing: {missing_pafs}')

    # format data
    for frame,detections in tqdm.tqdm(results.items(), disable=(not verbose)):
        if frame.startswith('frame'):
            frame_ix = int(frame[5:])
    
            idxs = []
            for k in range(K):
                if len(detections['confidence'][k][:,0]) > 0:
                    idx = np.argsort(detections['confidence'][k][:,0])[::-1][:N]
                    confidences[:len(idx),frame_ix,k] = detections['confidence'][k][idx,0]
                    identities[:len(idx),:,frame_ix,k] = detections['identity'][k][idx,:][:,id_channels]
                    coordinates[:len(idx),frame_ix,k] = detections['coordinates'][0][k][idx]
                    idxs.append(idx)
                else: idxs.append([])

            o = np.argsort(node_order)
            for ind,(k1,k2) in zip(metadata['PAFinds'],metadata['PAFgraph']):
                if len(idxs[k1])==0 or len(idxs[k2])==0: continue
                m1 = detections['costs'][ind]['m1']
                if o[k1] < o[k2]: m1,k1,k2 = m1.T,k2,k1
                affinities[:len(idxs[k1]),:len(idxs[k2]),frame_ix,k1] = m1[idxs[k1],:][:,idxs[k2]]

    return (coordinates[:,:,node_order],
            confidences[:,:,node_order],
            affinities[:,:,:,node_order],
            identities[:,:,:,node_order])


def format_deeplabcut_results(dataset_info, bodyparts, node_order, parents, 
                              individuals=None, parallelize=True, overwrite=False,
                              **kwargs):
    """
    Reformat DeepLabCut full.pickle files using `load_deeplabcut_full`.

    For each results path in `dataset_info`, the file
    `[path]_full.pickle` will be loaded and reformatted using
    `load_deeplabcut_full`. The results will be saved to a new file
    called `[path].h5` in the same directory.

    See `load_deeplabcut_full` for more details, including the
    parameters `bodyparts`, `node_order`, `parents`, and `individuals`.

    Parameters
    ----------
    dataset_info : dict
        Nested dictionary with the following structure::

            {
                'name_of_recording': {  
                    'keypoint_detections': { 
                        'name_of_camera': 'path/to/dlc_results',
                        'name_of_camera': 'path/to/dlc_results',
                        ...
                    },
                    'recorded_individuals': ['name_of_individual', ...]
                },
                ...
            }

    parallelize : bool, default: True
        If True, the reformatting will be parallelized across recordings.

    overwrite : bool, default: False
        If True, already extracted files will be overwritten. Otherwise
        they will be skipped.
    """
    def reformat(path, recorded_individuals):
        detections = load_deeplabcut_full(
            f'{path}_full.pickle', bodyparts, node_order, parents, 
            individuals, recorded_individuals, verbose=(not parallelize))
        save_detections_to_h5(f'{path}.h5', *detections)
        
    run_queue = []
    for recording_info in dataset_info.values():
        for path in recording_info['keypoint_detections'].values():
            if os.path.exists(f'{path}.h5') and not overwrite: continue
            run_queue.append((path, recording_info['recorded_individuals']))

    if parallelize:
        with multiprocessing.Pool() as pool:
            pool.starmap(reformat, run_queue)
    else:
        for path, recorded_individuals in run_queue:
            reformat(path, recorded_individuals)



def format_sleap_paf_graph(node_order, parents, n_individuals,
                           peaks, peak_vals, peak_channel_inds, 
                           edge_peak_inds, line_scores, **kwargs):
    """
    Format a SLEAP-inferred PAF graphs from a batch of images.

    - Bodyparts are reordered according to `node_order`.
    - The PAF graph is structured according to `parents`.
    - Only the top `n_individuals` of each peak is kept.

    Parameters
    ----------
    node_order : list of int
        Node order to be used during modeling.

    parents : list of int
        Child-parent relationships using the indexes from 
        `node_order`, such that `parent[i]==j` when `node_order[j]` 
        is the parent of `node_order[i]`.
        
    n_individuals : int
        Max instances of each peak to keep per image.

    peaks : ndarray of shape (n_frames, n_peaks, 2)
        x,y coordinates of all peaks.
    
    peak_vals : ndarray of shape (n_frames, n_peaks)
        Confidence values for each peak.

    peak_channel_inds : ndarray of shape (n_frames, n_peaks)
        Channel index (i.e. target bodypart) for each peak.

    edge_peak_inds : ndarray of shape (n_frames, n_edges, 2)
        Indices of peaks that form each edge in the graph.

    line_scores : ndarray of shape (n_frames, n_edges,)
        Confidence values for each edge in the graph.

    Returns
    -------
    coordinates: ndarray of shape (n_individuals, n_frames, n_bodyparts, 2)
        Coordinates of multi-animal keypoints detections from one or more cameras.
        
    confidences: ndarray of shape (n_individuals, n_frames, n_bodyparts)
        Confidences of multi-animal keypoints detections from one or more cameras.
        
    affinities: ndarray of shape (n_individuals, n_individuals, n_frames, n_bodyparts)
        Part affinity field weights for each edge in the node hierarchy,
        where `affinities[i,j,t,k]` is the weight from keypoint `[i,t,k]`
        to keypoint `[j,t,parent(k)]`.
    """
    # relabel peaks according to `node_order`
    peak_channel_inds = np.argsort(node_order)[np.maximum(peak_channel_inds,0)]

    # get dimensions
    T = peaks.shape[0]   # number of frames
    K = len(node_order)  # number of keypoints
    N = n_individuals    # number of individuals

    # initialize outputs
    coordinates = np.zeros((N,T,K,2))*np.nan
    confidences = np.zeros((N,T,K))
    affinities = np.zeros((N,N,T,K))

    # format data
    for t in range(T):
        idxs = []
        for k in range(K):
            idx = np.all([
                peak_channel_inds[t]==k, 
                ~np.isnan(peak_vals[t])
            ], axis=0).nonzero()[0]

            if len(idx) > 0:
                idx = idx[np.argsort(peak_vals[t][idx])[::-1]][:N]
                confidences[:len(idx),t,k] = peak_vals[t][idx]
                coordinates[:len(idx),t,k] = peaks[t][idx]
            idxs.append(idx)

        edges = {}
        for edge,s in zip(edge_peak_inds[t], line_scores[t]):
            if (edge >= 0).all():
                edge = edge[np.argsort(peak_channel_inds[t][edge])]
                edges[tuple(edge[::-1])] = s

        for child,parent in enumerate(parents):
            for i,peak_i in enumerate(idxs[child]):
                for j,peak_j in enumerate(idxs[parent]):
                    if (peak_i,peak_j) in edges:
                        affinities[i,j,t,child] = edges[(peak_i,peak_j)]

    return coordinates, confidences, affinities


def sleap_infer_paf_graphs(video_path, model_path, node_order, 
                           parents, n_individuals, 
                           peak_threshold=None, batch_size=None):
    """
    Infer PAF graphs from a video using a bottom-up SLEAP model.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    
    model_path : str
        Path to a bottom-up SLEAP model.

    node_order : list of int
        Node order to be used during modeling.

    parents : list of int
        Child-parent relationships using the indexes from 
        `node_order`, such that `parent[i]==j` when `node_order[j]` 
        is the parent of `node_order[i]`.
        
    n_individuals : int
        Max instances of each bodyparts to keep per image.

    peak_threshold : float, default=None
        Minimum confidence of a peak to be considered a valid detection.
        If None, the default threshold of the model will be used.

    batch_size : int, default=None
        Number of frames to process at a time. If None, the default
        batch size of the model will be used.

    Returns
    -------
    coordinates: ndarray of shape (n_individuals, n_frames, n_bodyparts, 2)
        Coordinates of multi-animal keypoint detections.
        
    confidences: ndarray of shape (n_individuals, n_frames, n_bodyparts)
        Confidences of multi-animal keypoints detections.
        
    affinities: ndarray of shape (n_individuals, n_individuals, n_frames, n_bodyparts)
        Part affinity field weights for each edge in the node hierarchy,
        where `affinities[i,j,t,k]` is the weight from keypoint `[i,t,k]`
        to keypoint `[j,t,parent(k)]`.
    """
    from sleap.nn.data.pipelines import VideoReader
    from sleap import load_model

    # load the sleap model
    predictor = load_model(model_path)
    predictor.inference_model.bottomup_layer.return_paf_graph = True

    if batch_size is not None:
        predictor.batch_size = batch_size
    if peak_threshold is not None:
        predictor.inference_model.bottomup_layer.peak_threshold = peak_threshold

    # initialize the video reader
    provider = VideoReader.from_filepath(filename=video_path)
    pipeline = predictor.make_pipeline(provider)

    # do inference
    tracking_results = []
    for batch in tqdm.tqdm(pipeline.make_dataset()):
        paf_graph = predictor.inference_model.predict(batch)
        tracking_results.append(format_sleap_paf_graph(
            node_order, parents, n_individuals, **paf_graph))
        
    coordinates = np.concatenate([t[0] for t in tracking_results], axis=1)
    confidences = np.concatenate([t[1] for t in tracking_results], axis=1)
    affinities = np.concatenate([t[2] for t in tracking_results], axis=2)
    return coordinates, confidences, affinities


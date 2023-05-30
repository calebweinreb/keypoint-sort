import os
import glob
import yaml
import tqdm
import h5py
import pickle
import numpy as np

import warnings
from textwrap import fill
warnings.formatwarning = lambda msg, *a: str(msg)

from .util import build_node_hierarchy


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
        'max_distance_std': 3,
        'min_confidence': 0.5,
        'num_sample_poses': 500,
        'num_sample_session': 10,
        'num_iters': 500,
        'num_states': 50,
        'num_animals': 2,
        'identity_pseudocount': .3,
        'outlier_pseudocount': 1e-5,
        'affinity_center': 0.2,
        'affinity_gain': 20,
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
     

def list_files(filepath_pattern, recursive):
    """
    This function lists all the files matching a pattern.

    Parameters
    ----------
    filepath_pattern : str or list
        A filepath pattern or a list thereof. Filepath patterns can be
        be a single file, a directory, or a path with wildcards (e.g.,
        '/path/to/dir/prefix*').

    recursive : bool, default=True
        Whether to search for files recursively.

    Returns
    -------
    list
        A list of file paths.
    """   
    if isinstance(filepath_pattern, list):
        matches = []
        for fp in filepath_pattern:
            matches += list_files(fp, recursive)
        return sorted(set(matches))
    
    else:
        matches = glob.glob(filepath_pattern)
        if recursive:
            for match in list(matches):
                matches += glob.glob(os.path.join(match, '**'), recursive=True)
        return matches
    

def list_files_with_exts(filepath_pattern, ext_list, recursive=True):
    """
    This function lists all the files matching a pattern and with a
    an extension/suffix in a list of extensions.

    Parameters
    ----------
    filepath_pattern : str or list
        A filepath pattern or a list thereof. Filepath patterns can be
        be a single file, a directory, or a path with wildcards (e.g.,
        '/path/to/dir/prefix*').

    ext_list : list of str
        A list of file extensions to search for.

    recursive : bool, default=True
        Whether to search for files recursively.

    Returns
    -------
    list
        A list of file paths.
    """
    ext_list = ['.'+ext.strip('.').lower() for ext in ext_list]
    has_ext = lambda f: os.path.splitext(f)[1].lower() in ext_list
    matches = list(filter(has_ext, list_files(filepath_pattern, recursive)))
    return matches
    

def list_files_with_suffixes(filepath_pattern, suffix_list, recursive=True):
    """
    This function lists all the files matching a pattern and with a
    a suffix in the specified list.

    Parameters
    ----------
    filepath_pattern : str or list
        A filepath pattern or a list thereof. Filepath patterns can be
        be a single file, a directory, or a path with wildcards (e.g.,
        '/path/to/dir/prefix*').

    suffix_list : list of str
        A list of acceptable suffixes.

    recursive : bool, default=True
        Whether to search for files recursively.

    Returns
    -------
    list
        A list of file paths.
    """
    has_suffix = lambda s: any([s.endswith(suff) for suff in suffix_list])
    matches = list(filter(has_suffix, list_files(filepath_pattern, recursive)))
    return matches


def _name_from_path(filepath, path_in_name, path_sep):
    """
    Create a name from a filepath. Either return the name of the file
    (with the extension removed) or return the full filepath, where the
    path separators are replaced with `path_sep`.
    """
    filepath = os.path.splitext(filepath)[0]
    if path_in_name:
        return filepath.replace(os.path.sep, path_sep)
    else:
        return os.path.basename(filepath)
    

def find_matching_videos(keys, video_dir, as_dict=False, recursive=True, 
                         session_name_suffix='', video_extension=None):
    """
    Find video files for a set of session names. The filename of each
    video is assumed to be a prefix within the session name, i.e. the
    session name has the form `{video_name}{more_text}`. If more than 
    one video matches a session name, the longest match will be used. 
    For example given the following video directory::

        video_dir
        ├─ videoname1.avi
        └─ videoname2.avi
 
    the videos would be matched to session names as follows::

        >>> keys = ['videoname1blahblah','videoname2yadayada']
        >>> find_matching_videos(keys, video_dir, as_dict=True)

        {'videoname1blahblah': 'video_dir/videoname1.avi',
         'videoname2blahblah': 'video_dir/videoname2.avi'}

    A suffix can also be specified, in which case the session name 
    is assumed to have the form `{video_name}{suffix}{more_text}`.
 
    Parameters
    -------
    keys: iterable
        Session names (as strings)

    video_dir: str
        Path to the video directory. 
        
    video_extension: str, default=None
        Extension of the video files. If None, videos are assumed to 
        have the one of the following extensions: "mp4", "avi", "mov"

    recursive: bool, default=True
        If True, search recursively for videos in subdirectories of
        `video_dir`.

    as_dict: bool, default=False
        Determines whether to return a dict mapping session names to 
        video paths, or a list of paths in the same order as `keys`.

    session_name_suffix: str, default=None
        Suffix to append to the video name when searching for a match.

    Returns
    -------
    video_paths: list or dict (depending on `as_dict`)
    """  

    if video_extension is None:
        extensions = ['.mp4','.avi','.mov']
    else: 
        if video_extension[0] != '.': 
            video_extension = '.'+video_extension
        extensions = [video_extension]

    videos = list_files_with_exts(video_dir, extensions, recursive=recursive)
    videos_to_paths = {os.path.splitext(os.path.basename(f))[0]:f for f in videos}

    video_paths = []
    for key in keys:
        matches = [v for v in videos_to_paths if \
                   os.path.basename(key).startswith(v+session_name_suffix)]
        assert len(matches)>0, fill(f'No matching videos found for {key}')
        
        longest_match = sorted(matches, key=lambda v: len(v))[-1]
        video_paths.append(videos_to_paths[longest_match])

    if as_dict: return dict(zip(sorted(keys),video_paths))
    else: return video_paths



def load_deeplabcut_detections(filepaths, bodyparts, node_order, parents, 
                               individuals=None, recorded_individuals=None, 
                               matched_frames=None, verbose=True, **kwargs):
    """
    Load multi-animal DeepLabCut results from a "_full.p" file.

    - Bodyparts are reordered according to `node_order`.
    - The PAF graph is restructured according to `parents`.
    - The number of individuals is set to `len(recorded_individuals)`
      if it is provided, otherwise to `len(individuals)`. All data
      is truncated and/or NaN padded to match this number.

    Parameters
    ----------
    filepaths : str or list of str
        Path(s) to the DeepLabCut results. For single-camera recordings,
        this can be a string or a list with a single element. For
        multi-camera recordings, this should be a list of strings. The
        order of the filepaths should match the camera-order in the
        calibration data that is used in subsequent processing.

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
        Subset of individuals in the current recorded. Will only be
        used if `individuals` is specified as well.

    matched_frames : ndarray of shape (num_frames,num_cameras), default=None
        Array of time-matched indices for each camera. For example
        `matched_frames[10,:]==[20,21]` would indicate that frame 20
        from camera 0 and frame 21 from camera 1 correspond to the
        were captured simultaneously. If None, it is assumed that
        the frames are already matched across cameras.
    
    verbose : bool, default=True
        If True, print status updates and show progress bars.

    Returns
    -------
    coordinates: ndarray of shape (n_individuals, n_frames, n_bodyparts, n_cameras, 2)
        Coordinates of multi-animal keypoints detections from one or more cameras.
        
    confidences: ndarray of shape (n_individuals, n_frames, n_bodyparts, n_cameras)
        Confidences of multi-animal keypoints detections from one or more cameras.
        
    affinities: ndarray of shape (n_individuals, n_individuals, n_frames, n_bodyparts, n_cameras)
        Part affinity field weights for each edge in the node hierarchy,
        where `affinities[i,j,t,k,c]` is the weight from keypoint `[i,t,k,c]`
        to keypoint `[j,t,parent(k),c]`.
            
    identities: ndarray of shape (n_individuals, n_individuals, n_frames, n_bodyparts, n_cameras)
        Identity weights for each keypoint, where `identities[i,j,t,k,c]`
        is the weight that keypoint `[i,t,k,c]` belongs to individual `j`.
    """
    # initialize `recorded_individuals`
    if recorded_individuals is None: 
        recorded_individuals = individuals
        id_channels = np.arange(len(individuals))
    else:
        id_channels = np.array([individuals.index(i) for i in recorded_individuals])

    # initialize `filepaths`
    if isinstance(filepaths, str): filepaths = [filepaths]

    # load data from each file
    if verbose: print('Loading DeepLabCut files')
    all_results = [pickle.load(open(fp, 'rb')) for fp in tqdm.tqdm(
        filepaths, desc='Loading', disable=(not verbose or len(filepaths)==1))]
    
    # initialize `matched_frames`
    num_frames = [r['metadata']['nframes'] for r in all_results]
    assert len(set(num_frames))==1 or matched_frames is not None, fill(
        'All videos must have the same number of frames unless `matched_frames` is provided')
    
    if matched_frames is not None:
        assert matched_frames.max(0) < np.array(num_frames), fill(
              '`matched_frames` contains indices that are out of range. '
            + f'The DeepLabCut metadata reports the following frame counts: {num_frames}, '
            + f'but the maximum indeces in `matched_frames` are {matched_frames.max(0)}')
          
    if matched_frames is None:
        matched_frames = np.repeat(np.arange(num_frames[0])[:,None],len(num_frames),axis=1)
    
    # get dimensions
    C = len(filepaths)            # number of cameras
    T = matched_frames.shape[0]   # number of frames
    K = len(bodyparts)            # number of keypoints
    N = len(recorded_individuals) # number of individuals 

    coordinates = np.zeros((N,T,K,C,2))*np.nan
    confidences = np.zeros((N,T,K,C))
    affinities = np.zeros((N,N,T,K,C))
    identities = np.zeros((N,N,T,K,C))
    
    if verbose: print('Formatting data')
    for c, (results, filepath) in enumerate(zip(all_results, filepaths)):
        metadata = results['metadata']

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
                        confidences[:len(idx),frame_ix,k,c] = detections['confidence'][k][idx,0]
                        identities[:len(idx),:,frame_ix,k,c] = detections['identity'][k][idx,:][:,id_channels]
                        coordinates[:len(idx),frame_ix,k,c] = detections['coordinates'][0][k][idx]
                        idxs.append(idx)
                    else: idxs.append([])

                o = np.argsort(node_order)
                for ind,(k1,k2) in zip(metadata['PAFinds'],metadata['PAFgraph']):
                    if len(idxs[k1])==0 or len(idxs[k2])==0: continue
                    m1 = detections['costs'][ind]['m1']
                    if o[k1] < o[k2]: m1,k1,k2 = m1.T,k2,k1
                    affinities[:len(idxs[k1]),:len(idxs[k2]),frame_ix,k1,c] = m1[idxs[k1],:][:,idxs[k2]]

    return (coordinates[:,:,node_order],
            confidences[:,:,node_order],
            affinities[:,:,:,node_order],
            identities[:,:,:,node_order])


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
                           parents, n_individuals):
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
    predictor.inference_model.bottomup_layer.peak_threshold = 0.02

    # initialize the video reader
    provider = VideoReader.from_filepath(filename=video_path)
    pipeline = predictor.make_pipeline(provider)

    # do inference
    tracking_results = []
    for batch in tqdm.tqdm(pipeline.make_dataset()):
        paf_graph = predictor.inference_model.predict(batch)
        tracking_results.append(format_sleap_paf_graph(
            node_order, parents, n_individuals, **paf_graph))
        
    coordinates, confidences, affinities = map(np.array, zip(*tracking_results))
    return coordinates, confidences, affinities


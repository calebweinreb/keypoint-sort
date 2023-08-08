import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2
import os
import tqdm
from scipy.ndimage import gaussian_filter1d
from .util import interpolate_keypoints


def plot_pose(ax, pose, parents=None, colors='k', window_size=150, 
              keypoints_to_show=None, set_bounds=True, flip_yaxis=True):
    """
    Plot a single pose on an axis using matplotlib.

    Parameters
    ----------
    ax : matplotlib axis
        Axis to plot on

    pose : ndarray, shape (n_keypoints, 2)
        Pose to plot

    parents : ndarray, shape (n_keypoints,), optional
        Parent of each keypoint

    colors : color as str or tuple, or a list of colors
        Single color or list of colors (one for each keypoint).

    window_size : int, optional
        Size of the window for cropping around the pose.

    set_bounds : bool, optional
        Whether to set the bounds of the axis to the window size.

    keypoints_to_show : list/array of int, optional
        Indices of keypoints to show. If None, all keypoints are shown.

    flip_yaxis : bool, optional
        Whether to flip the y-axis. This is useful for plotting poses
        in images, where the y-axis is flipped.
    """
    if not isinstance(colors, list): 
        colors = [colors]*len(pose)

    if keypoints_to_show is None:
        keypoints_to_show = np.arange(len(pose))
    
    if parents is not None:
        for i,j in enumerate(parents):
            if i in keypoints_to_show and j in keypoints_to_show:
                ax.plot(*pose[[i,j]].T, c=colors[i])
                
    colors = [colors[i] for i in keypoints_to_show]
    ax.scatter(*pose[keypoints_to_show].T, c=colors)

    if set_bounds:
        cen = np.median(pose, axis=0)
        xlim = [cen[0]-window_size//2,cen[0]+window_size//2]
        ylim = [cen[1]-window_size//2,cen[1]+window_size//2]
        if flip_yaxis: ylim = ylim[::-1]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    elif flip_yaxis:
        ylim = ax.get_ylim()
        ax.set_ylim(sorted(ylim)[::-1])



def crop_image(image, centroid, crop_size):
    """
    Crop an image around a centroid.

    Parameters
    ----------
    image: ndarray of shape (height, width, 3)
        Image to crop.

    centroid: tuple of int
        (x,y) coordinates of the centroid.

    crop_size: int or tuple(int,int)
        Size of the crop around the centroid. Either a single int for
        a square crop, or a tuple of ints (w,h) for a rectangular crop.

    Returns
    -------
    image: ndarray of shape (crop_size, crop_size, 3)
        Cropped image.
    """
    if isinstance(crop_size,tuple): w,h = crop_size
    else: w,h = crop_size,crop_size
    x,y = int(centroid[0]),int(centroid[1])

    x_min = max(0, x - w//2)
    y_min = max(0, y - h//2)
    x_max = min(image.shape[1], x + w//2)
    y_max = min(image.shape[0], y + h//2)

    cropped = image[y_min:y_max, x_min:x_max]
    padded = np.zeros((h,w,*image.shape[2:]), dtype=image.dtype)
    pad_x = (w - cropped.shape[1]) // 2
    pad_y = (h - cropped.shape[0]) // 2
    padded[pad_y:pad_y+cropped.shape[0], pad_x:pad_x+cropped.shape[1]] = cropped
    return padded




def overlay_keypoints_on_image(
    image, coordinates, parents, keypoint_colormap='autumn',
    node_size=2, line_width=1, copy=False, opacity=1.0):
    """
    Overlay keypoints on an image using OpenCV.

    Parameters
    ----------
    image: ndarray of shape (height, width, 3)
        Image to overlay keypoints on.
    
    coordinates: ndarray of shape (num_keypoints, 2) or list thereof
        Array of keypoint coordinates or list of such arrays
        (one per animal).

    parents : ndarray, shape (n_keypoints,), optional
        Parent of each keypoint 

    keypoint_colormap: list or str, default='autumn'
        Name of a matplotlib colormap to use for coloring the keypoints,
        or a list of colormaps (one for each animal).

    node_size: int, default=10
        Size of the keypoints.

    line_width: int, default=2
        Width of the skeleton lines.

    copy: bool, default=False
        Whether to copy the image before overlaying keypoints.
    
    opacity: float, default=1.0
        Opacity of the overlay graphics (0.0-1.0).

    Returns
    -------
    image: ndarray of shape (height, width, 3)
        Image with keypoints overlayed.
    """
    if copy or opacity<1.0: 
        canvas = image.copy()
    else: canvas = image

    if not isinstance(keypoint_colormap, list):
        keypoint_colormap = [keypoint_colormap]
    
    if not isinstance(coordinates, list):
        coordinates = [coordinates]

    def is_valid(x,y):
        h,w = image.shape[:2]
        return not np.any([np.isnan(x), np.isnan(y), x<0, y<0, x>=w, y>=h])
    
    for coords, cmap in zip(coordinates, keypoint_colormap):
        # get colors from matplotlib and convert to 0-255 range for openc
        colors = plt.get_cmap(cmap)(np.linspace(0,1,coords.shape[0]))
        colors = [tuple([int(c) for c in cs[:3]*255]) for cs in colors]

        # overlay skeleton
        for i, j in enumerate(parents):
            if i != j and is_valid(*coords[i]) and is_valid(*coords[j]):
                pos1 = (int(coords[i][0]), int(coords[i][1]))
                pos2 = (int(coords[j][0]), int(coords[j][1]))
                canvas = cv2.line(canvas, pos1, pos2, colors[i], line_width, cv2.LINE_AA)

        # overlay keypoints
        for i, (x,y) in enumerate(coords):
            if is_valid(x,y):
                canvas = cv2.circle(
                    canvas, (int(x), int(y)), node_size, 
                    colors[i], -1, lineType=cv2.LINE_AA)

    if opacity<1.0:
        image = cv2.addWeighted(image, 1-opacity, canvas, opacity, 0)
    return image


def overlay_keypoints_on_video(
    video_path, coordinates, parents, output_path=None, 
    keypoint_colormap='autumn', node_size=2, line_width=1,
    show_frame_numbers=True, text_color=(255,255,255), 
    crop_size=None, frames=None, quality=7, 
    centroid_smoothing_filter=10):
    """
    Overlay keypoints on a video using OpenCV.

    Parameters
    ----------
    video_path: str
        Path to a video file.

    coordinates: ndarray of shape (num_frames, num_keypoints, 2) or list thereof
        Array of keypoint coordinates or list of such arrays
        (one per animal).

    parents : ndarray, shape (n_keypoints,), optional
        Parent of each keypoint 

    output_path: str, default=None
        Path to save the video. If None, the video is saved to
        `video_path` with the suffix `_overlay`.

    keypoint_colormap: list or str, default='autumn'
        Name of a matplotlib colormap to use for coloring the keypoints,
        or a list of colormaps (one for each animal).

    node_size: int, default=10
        Size of the keypoints.

    line_width: int, default=2
        Width of the skeleton lines.

    show_frame_numbers: bool, default=True
        Whether to overlay the frame number in the video.

    text_color: tuple of int, default=(255,255,255)
        Color for the frame number overlay.

    crop_size: int, default=None
        Size of the crop around the keypoints to overlay on the video.
        If None, the entire video is used. 

    frames: iterable of int, default=None
        Frames to overlay keypoints on. If None, all frames are used.

    quality: int, default=7
        Quality of the output video.

    centroid_smoothing_filter: int, default=10
        Amount of smoothing to determine cropping centroid.

    plot_options: dict, default={}
        Additional keyword arguments to pass to
        :py:func:`keypoint_moseq.viz.overlay_keypoints`.
    """
    if not isinstance(coordinates, list):
        coordinates = [coordinates]

    if not isinstance(keypoint_colormap, list):
        keypoint_colormap = [keypoint_colormap]

    if frames is None:
        frames = np.arange(coordinates[0].shape[0])

    if output_path is None: 
        output_path = os.path.splitext(video_path)[0] + '_overlay.mp4'

    if crop_size is not None:
        coords = np.concatenate(coordinates, axis=2)
        outliers = np.any(np.isnan(coords), axis=2)
        interpolated_coords = interpolate_keypoints(coords, outliers, axis=0)
        crop_centroid = np.nanmedian(interpolated_coords, axis=1)
        crop_centroid = gaussian_filter1d(crop_centroid, centroid_smoothing_filter, axis=0)

    with imageio.get_reader(video_path) as reader:
        fps = reader.get_meta_data()['fps']

        with imageio.get_writer(
            output_path, pixelformat='yuv420p', 
            fps=fps, quality=quality) as writer:

            for frame in tqdm.tqdm(frames):
                image = reader.get_data(frame)

                for coords, cmap in zip(coordinates, keypoint_colormap):
                    image = overlay_keypoints_on_image(
                        image, coords[frame], parents=parents, keypoint_colormap=cmap,
                        node_size=node_size, line_width=line_width)

                if crop_size is not None:
                    image = crop_image(image, crop_centroid[frame], crop_size)

                if show_frame_numbers:
                    image = cv2.putText(
                        image, f'Frame {frame}', (10, image.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, text_color, 1, cv2.LINE_AA)

                writer.append_data(image)


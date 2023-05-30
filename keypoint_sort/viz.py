import numpy as np
import matplotlib.pyplot as plt
import pickle, os

def plot_poses(poses, parents, colors=None, dims=(2,5), 
               window_size=150, inches_per_plot=2):
    
    num_axs = dims[0]*dims[1]
    num_kps = poses.shape[1]
    
    if poses.shape[0] <= num_axs: ixs = np.arange(num_axs)
    else: ixs = np.random.choice(poses.shape[0], size=num_axs, replace=False)
    if colors is None: colors = [plt.cm.jet(i/(num_kps-1)) for i in range(num_kps)]
    
    fig,axs = plt.subplots(*dims)
    for ax,ix in zip(axs.flat,ixs):
        for i,j in enumerate(parents):
            ax.plot(*poses[ix,[i,j]].T, c='k')
        ax.scatter(*poses[ix].T, c=colors)
        cen = (poses[ix].min(0)+poses[ix].max(0))/2
        ax.set_xlim([cen[0]-window_size//2,cen[0]+window_size//2])
        ax.set_ylim([cen[1]-window_size//2,cen[1]+window_size//2])    
        ax.set_xticks([]); ax.set_yticks([])
    fig.set_size_inches((dims[1]*inches_per_plot, dims[0]*inches_per_plot))
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 19:03:38 2022

@author: Amin
"""

from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.pyplot as plt
import numpy as np

# %%
def visualize_image(data,p,scale=1,microns=1,save=False,file=None,factor=5):
    plt.figure(figsize=(18,9))
    plt.imshow(factor*data[:,:,:,[0,1,2]].max(2)/data.max())
    c = p[:,3:6]/p[:,3:6].max()
    c[c<0] = 0
    P = p[:,:3].copy()*scale
    plt.scatter(P[:,1],P[:,0],c=c,marker='o',edgecolor='w')
    plt.axis('off')
    scalebar = ScaleBar(microns,'um')
    plt.gca().add_artist(scalebar)
    
    if save:
        try:
            plt.savefig(file+'.png',format='png')
            plt.savefig(file+'.pdf',format='pdf')
        except:
            pass
        plt.close('all')
    else:
        plt.pause(.1)
        plt.show()
        
# %%
def visualize_pc(aligned,title_str='',fontsize=9,dotsize=30,save=False,file=None):
    
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_title(title_str)

    for j in range(len(aligned)):
        al = aligned[j].T
        c_j = al[:,3:6]
        c_j[c_j < 0] = 0
        ax.scatter(al[:,0],al[:,1],al[:,2], s=10, c=c_j/c_j.max(),marker='.');
    
    axis_equal_3d(ax)
    ax.view_init(elev=90., azim=10)
    
    ax.axis('off')
    ax.set_facecolor('xkcd:light gray')
    fig.patch.set_facecolor('xkcd:light gray')
    
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()

# %%
def axis_equal_3d(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


# %%
def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)


def multi_slice_viewer(volume):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index])
    fig.canvas.mpl_connect('key_press_event', process_key)

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])


# %%
def visualize_subjects(imgs_):
    for i in range(len(imgs_)):
        plt.subplot(1,len(imgs_),i+1)
        plt.imshow(imgs_[i].mean(2))
        plt.title('Macaque '+ str(i+1))
        
# %%
def plot_loss(losses):
    plt.plot(losses)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
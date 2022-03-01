# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 19:03:38 2022

@author: Amin
"""

from matplotlib_scalebar.scalebar import ScaleBar
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import scipy as sp
# %%
def visualize_image(data,p=None,c=None,names=None,olp=False,tol=1e-3,scale=1,microns=1,titlestr='',fontsize=12,save=False,file=None,factor=5):
    plt.figure(figsize=(18,9))
    plt.imshow(factor*data.max(2)/data.max())
    
    if p is not None:
        c[c<0] = 0
        c = c/c.max()
        P = p.copy()*scale
        pos = optimal_label_positioning(P,lambda_1=10,lambda_2=10,tol=tol) if olp else P
        plt.scatter(P[:,1],P[:,0],s=100,c=c,marker='o',edgecolor='w')
        if names is not None:
            for n in range(len(names)):
                # c[n,:]
                plt.text(pos[n,1],pos[n,0],names[n],color='w',fontsize=fontsize)
                plt.plot([pos[n,1], P[n,1]],[pos[n,0], P[n,0]],color='w',linestyle='dotted',linewidth=1)

    
    plt.axis('off')
    scalebar = ScaleBar(microns,'um')
    plt.gca().add_artist(scalebar)
    
    plt.title(titlestr,fontsize=fontsize)
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()
        
# %%
def visualize_pc(aligned,atlas=None,names=None,title_str='',olp=False,fontsize=9,dotsize=30,save=False,file=None):
    
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_title(title_str)

    for j in range(len(aligned)):
        al = aligned[j].T
        c_j = al[:,3:6]
        c_j[c_j < 0] = 0
        ax.scatter(al[:,0],al[:,1],al[:,2], s=10, c=c_j/c_j.max(),marker='.')
        
    if atlas is not None:
        c = atlas[:,3:]
        c[c<0] = 0
        c = c/c.max()
        ax.scatter(atlas[:,0],atlas[:,1],atlas[:,2], 
               s=dotsize,edgecolor=c,facecolor='w',marker='o')
        
        name_pos = optimal_label_positioning(atlas[:,:3],lambda_1=1,lambda_2=1,tol=1e-1) if olp else atlas[:,:3]
        if names is not None:
            for i in range(len(names)):
                ax.text(name_pos[i,0],name_pos[i,1],name_pos[i,2],names[i],c=c[i,:])
                ax.plot([name_pos[i,0], atlas[i,0]],
                        [name_pos[i,1], atlas[i,1]],
                        [name_pos[i,2], atlas[i,2]],
                        color=c[i,:],linestyle='dotted',linewidth=1)
                
    ax.dist=5
    axis_equal_3d(ax)
    ax.view_init(elev=20., azim=0)
    
    ax.axis('off')
    ax.set_facecolor('xkcd:light gray')
    fig.patch.set_facecolor('xkcd:light gray')
    
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()


def optimal_label_positioning(mu,lambda_1=5,lambda_2=5,tol=1e0):
    n = mu.shape[0]
    D = sp.spatial.distance.squareform(sp.spatial.distance.pdist(mu))
    ll = np.random.rand(n,1)
    ll = lambda_1*ll*ll.T + lambda_2
    L = np.vstack((
        np.hstack((D,D+ll-np.diag(np.diag(D+ll))+np.diag(ll))),
        np.hstack((D+ll-np.diag(np.diag(D+ll))+np.diag(ll),D-np.diag(np.diag(D+ll))+2*ll))
        ))
    K = 1/(L**2)
    k = 2*K
    l = L
    
    myfun = lambda x: kk_cost(mu,x.reshape(mu.shape),k,l)
    res = minimize(myfun, mu.reshape(-1), method='L-BFGS-B', tol=tol)
    coor = res.x.reshape(mu.shape)
    return coor

def kk_cost(mu,coor,k,l):
    y = np.vstack((mu,coor))
    
    pdist = sp.spatial.distance.squareform(sp.spatial.distance.pdist(y))
    cost = np.triu(k*(pdist-l)**2,1).sum()
    
    return cost
# %%
def axis_equal_3d(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/4
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
def visualize_subjects(imgs_,titles,save=False,file=None):
    m = int(np.sqrt(len(imgs_)))
    n = np.ceil(len(imgs_)/m)
    plt.figure(figsize=(15,9))
    for i in range(len(imgs_)):
        plt.subplot(m,n,i+1)
        plt.imshow(imgs_[i].mean(2))
        plt.title(titles[i])
        plt.axis('off')
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()

# %%
def plot_loss(losses,labels=None,titlestr='',fontsize=12,save=False,file=None):
    plt.figure(figsize=(10,5))
    
    colors = plt.cm.hsv(np.linspace(0,1,len(losses)+1)[0:-1])[:,0:3]

    for i in range(len(losses)):
        plt.plot(losses[i],color=colors[i],lw=2,label=labels[i])
        
    plt.yscale('log')
    plt.grid('on')
    plt.xlabel('Iterations',fontsize=fontsize)
    plt.ylabel('Loss',fontsize=fontsize)
    plt.title(titlestr,fontsize=fontsize)
    
    handles, labels = plt.gca().get_legend_handles_labels()

    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    plt.legend(handles, labels, loc='best')

    
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()
        
# %%
def plot_bar(val,fontsize=12,titlestr='',ylabel='',ticks=None,save=False,file=None):
    plt.figure(figsize=(len(val)*2,5))
    
    colors = plt.cm.hsv(np.linspace(0,1,len(val)+1)[0:-1])[:,0:3]
    
    ax = plt.boxplot(val,patch_artist=True)
    
    for i in range(len(ax['boxes'])):
        ax['boxes'][i].set(facecolor=colors[i])


    # plt.errorbar(np.arange(len(val)), [a.mean() for a in val], yerr=[stats.sem(a)  for a in val],
    #                    color='k',lw=2)
    plt.xticks(np.arange(1,len(val)+1),ticks,fontsize=fontsize)
    plt.grid('on')
    plt.title(titlestr,fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
    
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()


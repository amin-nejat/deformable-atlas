# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 19:03:38 2022

@author: Amin
"""
# %%
import matplotlib.pyplot as plt

import numpy as np
import scipy as sp

from scipy import stats
from scipy.optimize import minimize
# %%
def visualize_3d(data,p=None,c=None,factor=2,save=False,file=None):
    '''Visualize volumetric image by max projecting on xy, yz, zx.
    '''
    orders = [[2,1,0],[0,1,2],[2,0,1]]
    if c is not None:
        c[c<0] = 0
        c = c/c.max()
    
    plt.figure(figsize=(20,10))
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=3)
    ax2 = plt.subplot2grid((2, 3), (1, 0), colspan=2, rowspan=1)
    ax3 = plt.subplot2grid((2, 3), (1, 2), rowspan=1)

    ax = [ax1,ax2,ax3]
    
    for i,o in enumerate(orders):
        img = data.transpose(o[0],o[1],o[2],3)
        if p is not None:
            P = p[:,o]
            ax[i].scatter(P[:,1],P[:,0],s=10,c=c,marker='o',edgecolor='w')
        ax[i].imshow(factor*img.max(2)/img.max())
        ax[i].axis('off')
        
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()
        
        
# %%
def visualize_image(
        data,t_stat=None,
        p=None,c=None,names=None,olp=False,tol=1e-3,
        xlim=None,ylim=None,
        scale=1,microns=1,titlestr='',fontsize=12,factor=5,
        save=False,file=None
    ):
    '''Visualize 2d max projected image along with corresponding point clouds
    '''
    plt.figure(figsize=(18,9))
    if t_stat is None:
        plt.imshow(factor*data.max(2)/data.max())
    else:
        norm_t_stat = (t_stat-t_stat.min())/((t_stat.max()-t_stat.min()))
        plt.imshow(factor*data.max(2)/data.max())
        img = plt.imshow(
            norm_t_stat.squeeze(), 
            alpha=norm_t_stat.squeeze(),
            cmap='jet',
        )
        ax = plt.gca()
        plt.colorbar(img,ax=ax)

    
    if p is not None:
        c[c<0] = 0
        c = c/c.max()
        P = p.copy()*scale
        pos = optimal_label_positioning(P,lambda_1=10,lambda_2=10,tol=tol) if olp else P
        plt.scatter(P[:,1],P[:,0],s=10,c=c,marker='o',edgecolor='w')
        if names is not None:
            for n in range(len(names)):
                # c[n,:]
                plt.text(pos[n,1],pos[n,0],names[n],color='w',fontsize=fontsize)
                plt.plot([pos[n,1], P[n,1]],[pos[n,0], P[n,0]],color='w',linestyle='dotted',linewidth=1)

    if xlim is not None: plt.xlim(xlim)
    if ylim is not None: plt.ylim(ylim)
    
    plt.axis('off')
    plt.gca().add_artist(scalebar)
    
    plt.title(titlestr,fontsize=fontsize)
    if save:
        plt.savefig(file+'.png',format='png')
        try: plt.savefig(file+'.pdf',format='pdf')
        except: pass
        plt.close('all')
    else:
        plt.show()
        
# %%
def visualize_pc(
        aligned,atlas=None,names=None,three_d=False,
        title_str='',olp=False,fontsize=9,dotsize=30,
        save=False,file=None
    ):
    '''Visualize point clouds and atlases learned from them
    '''
    fig = plt.figure(figsize=(15,8))
    
    
    if three_d: 
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.gca()
    
    ax.set_title(title_str)

    for j in range(len(aligned)):
        al = aligned[j].T
        c_j = al[:,3:6]
        c_j[c_j < 0] = 0
        if three_d:
            ax.scatter(al[:,0],al[:,1],al[:,2], s=10, c=c_j/c_j.max(),marker='.')
        else:
            ax.scatter(al[:,1],al[:,0], s=10, c=c_j/c_j.max(),marker='.')
        
    if atlas is not None:
        c = atlas[:,3:]
        c[c<0] = 0
        c = c/c.max()
        if three_d:
            ax.scatter(
                atlas[:,0],atlas[:,1],atlas[:,2], 
                s=dotsize,edgecolor=c,facecolor='w',marker='o'
            )
        else:
            ax.scatter(
                atlas[:,1],atlas[:,0],s=dotsize,edgecolor=c,
                facecolor='w',marker='o'
            )
            
        
        name_pos = optimal_label_positioning(atlas[:,:3],lambda_1=1,lambda_2=1,tol=1e-1) if olp else atlas[:,:3]
        if names is not None:
            for i in range(len(names)):
                if three_d:
                    ax.text(name_pos[i,0],name_pos[i,1],name_pos[i,2],names[i],c=c[i,:])
                    ax.plot([name_pos[i,0], atlas[i,0]],
                            [name_pos[i,1], atlas[i,1]],
                            [name_pos[i,2], atlas[i,2]],
                            color=c[i,:],linestyle='dotted',linewidth=1)
                else:
                    ax.text(name_pos[i,1],name_pos[i,0],names[i],c=c[i,:])
                    ax.plot([name_pos[i,1], atlas[i,1]],
                            [name_pos[i,0], atlas[i,0]],
                            color=c[i,:],linestyle='dotted',linewidth=1)
                    
                    draw_ellipse(
                        atlas[i,[1,0]],np.cov(aligned[:,[1,0],i].T),c[i],
                        std_devs=1,ax=ax,line_width=1
                    )
                
    ax.dist=5
    if three_d: 
        axis_equal_3d(ax)
        ax.view_init(elev=20., azim=0)
    else:
        ax.axis('equal')
    
    
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
    '''Using an optimization algorithm to position labels in a non-overlapping
        arrangement.
    '''
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
    '''Cost used for the optimization of the figure labels.
    '''
    y = np.vstack((mu,coor))
    pdist = sp.spatial.distance.squareform(sp.spatial.distance.pdist(y))
    cost = np.triu(k*(pdist-l)**2,1).sum()
    
    return cost
# %%
def axis_equal_3d(ax):
    '''Equalizing 3D plot axes
    '''
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/4
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


# %%
def remove_keymap_conflicts(new_keys_set):
    '''Helper for multi-slice viewer.
    '''
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def multi_slice_viewer(volume):
    '''Visualizing volumetric images by viewing cross sections using keys 'j'
        and 'k'
    '''
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index])
    fig.canvas.mpl_connect('key_press_event', process_key)

def process_key(event):
    '''Helper for multi-slice viewer.
    '''
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    '''Helper for multi-slice viewer.
    '''
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])

def next_slice(ax):
    '''Helper for multi-slice viewer.
    '''
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])


# %%
def visualize_subjects(imgs_,titles,cmap='gray',factor=1,save=False,file=None):
    '''Visualize multiple images in subplots.
    '''
    m = int(np.sqrt(len(imgs_)))
    n = np.ceil(len(imgs_)/m)
    plt.figure(figsize=(n*6,m*5))
    for i in range(len(imgs_)):
        plt.subplot(m,n,i+1)
        im = imgs_[i].mean(2)
        if len(im.shape) == 3:
            plt.imshow(factor*im[:,:,:3]/im[:,:,:3].max())
        else:
            plt.imshow(im,cmap=cmap)
        plt.title(titles[i])
        plt.axis('off')
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()

# %%
def plot_loss(losses,labels=None,titlestr='',fontsize=12,yscale='log',save=False,file=None):
    '''Plotting the training loss through iterations.
    '''
    plt.figure(figsize=(10,5))
    
    colors = plt.cm.hsv(np.linspace(0,1,len(losses)+1)[0:-1])[:,0:3]

    for i in range(len(losses)):
        plt.plot(losses[i],color=colors[i],lw=2,label=labels[i],alpha=.5)
        
    plt.yscale(yscale)
    plt.grid('on')
    plt.xlabel('Iterations',fontsize=fontsize)
    plt.ylabel('Loss',fontsize=fontsize)
    plt.title(titlestr,fontsize=fontsize)
    
    handles, labels = plt.gca().get_legend_handles_labels()

    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    plt.legend(handles, labels, loc='best', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()
        
# %%
def plot_bar(val,cmp,gp_names=['1','2'],fontsize=12,titlestr='',ylabel='',ticks=None,save=False,file=None):
    '''Bar plots for comparing a statistic across different groups.
    '''
    plt.figure(figsize=(len(val)*1.6,5))
    # colors = plt.cm.hsv(np.linspace(0,1,len(val)+1)[0:-1])[:,0:3]
    barwidth = .3
    plt.bar(
        np.arange(1,len(val)+1)-barwidth/2, 
        np.array(val).mean(1), 
        width = barwidth, 
        color = 'blue', 
        edgecolor = 'black', 
        yerr=np.array(val).std(1)/np.sqrt(len(val)), 
        capsize=7, label=gp_names[0]
    )
    
    plt.bar(
        np.arange(1,len(cmp)+1)+barwidth/2, 
        np.array(cmp).mean(1), 
        width = barwidth, 
        color = 'cyan', 
        edgecolor = 'black', 
        yerr=np.array(cmp).std(1)/np.sqrt(len(cmp)), 
        capsize=7, label=gp_names[1]
    )
    
    plt.legend()

    plt.xticks(np.arange(1,len(val)+1),ticks,fontsize=fontsize)
    plt.title(titlestr,fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()

# %%
def visualize_t(atlas,a,b,titlestr='',fontsize=15,save=False,file=None):
    '''Visualizing t-statistic of two populations super-imposed on the atlas.
    '''
    plt.figure(figsize=(10,10))

    t,p = stats.ttest_rel(a,b,axis=2)
    
    t[np.isnan(t)] = 0
    p[np.isnan(p)] = 1
    
    plt.imshow(t)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=fontsize)
    
    from statsmodels.stats import multitest
    rejected,_ = multitest.fdrcorrection(p.flatten(),alpha=.05)
    rejected = np.array(rejected).reshape(p.shape).astype(float)
    rejected[atlas < 0.1] = 0
    
    plt.imshow(atlas, cmap='gray')
    plt.imshow(t,alpha=rejected, cmap='viridis')
    plt.axis('off')
    
    plt.title(titlestr,fontsize=fontsize)
    
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()


# %%

from scipy.stats import multivariate_normal

def draw_ellipse(mean,covariance,colors,std_devs=3,ax=None,line_width=2):
    '''sample grid that covers the range of points'''
    min_p = mean - std_devs*np.sqrt(np.diag(covariance))
    max_p = mean + std_devs*np.sqrt(np.diag(covariance))
    
    x = np.linspace(min_p[0],max_p[0],256) 
    y = np.linspace(min_p[1],max_p[1],256)
    X,Y = np.meshgrid(x,y)
    
    Z = multivariate_normal.pdf(np.stack((X.reshape(-1),Y.reshape(-1))).T, mean=mean, cov=(std_devs**2)*covariance)
    Z = Z.reshape([len(x),len(y)])
    
    if ax is None:
        plt.contour(X, Y, Z, 0,colors=colors,linewidth=line_width)
    else:
        ax.contour(X, Y, Z, 0,colors=colors,linewidths=line_width)

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 12:30:50 2022

@author: Amin
"""
import numpy as np
import scipy as sp
import torch
import cv2

# %%
def image_iwarp(im,flow,grid):
    '''Inverting a warping operation.
    '''
    X = np.array([flow[:,:,:,0].reshape(-1), flow[:,:,:,1].reshape(-1), flow[:,:,:,2].reshape(-1)]).T
    Y = im.reshape(-1)
    interp = sp.interpolate.NearestNDInterpolator(X,Y)
    mapped = interp(grid).reshape(im.shape)
    return mapped

# %%
def image_warp(im, flow):
    '''Warping volumetric image according to a flow field
    '''
    
    return cv2.remap(im.numpy(), flow.astype(np.float32), None, cv2.INTER_NEAREST)[:,:,None]


# %%
def rgb2hsv(rgb):
    ''' R, G and B input range = 0 ÷ 1.0
        H, S and V output range = 0 ÷ 1.0
    '''
    eps = 1e-7

    var_R = rgb[:,0,:,:]
    var_G = rgb[:,1,:,:]
    var_B = rgb[:,2,:,:]

    var_Min = rgb.min(1)[0]    #Min. value of RGB
    var_Max = rgb.max(1)[0]    #Max. value of RGB
    del_Max = var_Max - var_Min             ##Delta RGB value

    H = torch.zeros([rgb.shape[0], rgb.shape[2], rgb.shape[3]]).to(rgb.device)
    S = torch.zeros([rgb.shape[0], rgb.shape[2], rgb.shape[3]]).to(rgb.device)
    V = torch.zeros([rgb.shape[0], rgb.shape[2], rgb.shape[3]]).to(rgb.device)

    V = var_Max

    #Chromatic data...
    S = del_Max / (var_Max + eps)

    del_R = ( ( ( var_Max - var_R ) / 6 ) + ( del_Max / 2 ) ) / (del_Max + eps)
    del_G = ( ( ( var_Max - var_G ) / 6 ) + ( del_Max / 2 ) ) / (del_Max + eps)
    del_B = ( ( ( var_Max - var_B ) / 6 ) + ( del_Max / 2 ) ) / (del_Max + eps)

    H = torch.where( var_R == var_Max , del_B - del_G, H)
    H = torch.where( var_G == var_Max , ( 1 / 3 ) + del_R - del_B, H)
    H = torch.where( var_B == var_Max ,( 2 / 3 ) + del_G - del_R, H)

    return torch.stack([H, S, V], 1)

# %%
def hsv2rgb(hsv):
    ''' H, S and V input range = 0 ÷ 1.0
        R, G and B output range = 0 ÷ 1.0
    '''

    bb,cc,hh,ww = hsv.shape
    H = hsv[:,0,:,:]
    S = hsv[:,1,:,:]
    V = hsv[:,2,:,:]

    var_h = H * 6

    var_h[var_h == 6] = 0      #H must be < 1
    var_i = var_h.floor()                           #Or ... var_i = floor( var_h )
    var_1 = V * ( 1 - S )
    var_2 = V * ( 1 - S * ( var_h - var_i ) )
    var_3 = V * ( 1 - S * ( 1 - ( var_h - var_i ) ) )

    # else                   { var_r = V     ; var_g = var_1 ; var_b = var_2 }
    var_r = V
    var_g = var_1
    var_b = var_2

    # var_i == 0 { var_r = V     ; var_g = var_3 ; var_b = var_1 }
    var_r = torch.where(var_i == 0, V, var_r)
    var_g = torch.where(var_i == 0, var_3, var_g)
    var_b = torch.where(var_i == 0, var_1, var_b)

    # else if ( var_i == 1 ) { var_r = var_2 ; var_g = V     ; var_b = var_1 }
    var_r = torch.where(var_i == 1, var_2, var_r)
    var_g = torch.where(var_i == 1, V, var_g)
    var_b = torch.where(var_i == 1, var_1, var_b)

    # else if ( var_i == 2 ) { var_r = var_1 ; var_g = V     ; var_b = var_3 }
    var_r = torch.where(var_i == 2, var_1, var_r)
    var_g = torch.where(var_i == 2, V, var_g)
    var_b = torch.where(var_i == 2, var_3, var_b)

    # else if ( var_i == 3 ) { var_r = var_1 ; var_g = var_2 ; var_b = V     }
    var_r = torch.where(var_i == 3, var_1, var_r)
    var_g = torch.where(var_i == 3, var_2, var_g)
    var_b = torch.where(var_i == 3, V, var_b)

    # else if ( var_i == 4 ) { var_r = var_3 ; var_g = var_1 ; var_b = V     }
    var_r = torch.where(var_i == 4, var_3, var_r)
    var_g = torch.where(var_i == 4, var_1, var_g)
    var_b = torch.where(var_i == 4, V, var_b)


    R = var_r #* 255
    G = var_g #* 255
    B = var_b #* 255

    return torch.stack([R, G, B], 1)

# %%
def sort_mu(ims,neurons=None,min_counts=2):
    '''Sorting the indices of multiple point clouds into an aggregated N.6.B
        numpy array where N is the number of points in the point cloud, 6 corresponds
        to (XYZ RGB), and B is the number of subjects.
    '''
    
    annotations = [x.get_annotations() for x in ims]
    scales = [np.array([1,1,1]) for x in ims]
    positions = [x.get_positions(x.scale) for x in ims]
    colors = [x.get_colors_readout() for x in ims]
    
    N = list(set([item for sublist in annotations for item in sublist]))
    
    C = colors[0].shape[1]
    pos = np.zeros((len(N),3,len(annotations)))*np.nan
    col = np.zeros((len(N),C,len(annotations)))*np.nan
    
    for j in range(len(annotations)):
        perm = np.array([N.index(x) for x in annotations[j]])
        pos[perm,:,j] = positions[j]*scales[j][None,:]
        col_tmp = colors[j]
        col[perm,:,j] = col_tmp
    
    counts = (~np.isnan(pos.sum(1))).sum(1)
    good_indices = np.logical_and( counts>=min_counts, 
                                  ~np.array([x == '' or x == None for x in N]))
    pos = pos[good_indices ,:,:]
    col = col[good_indices ,:,:]
    
    N = [N[i] for i in range(len(good_indices)) if good_indices[i]]
    
    if neurons is not None:
        idx = [i for i in range(len(N)) if N[i] in neurons]
        N = [N[i] for i in idx]
        pos = pos[idx,:]
        col = col[idx,:]
    
    return N,col,pos,counts

# %% helper functions for 3D quadratic basis and jacobian
def quadratic_basis(P):
    return torch.cat((P[...,0][...,None]*0,P,P*P,
                      P[...,0][...,None]*P[...,1][...,None],
                      P[...,0][...,None]*P[...,2][...,None],
                      P[...,1][...,None]*P[...,2][...,None]),len(P.shape)-1)

def quadratic_det_jac(B,P):
    x,y,z = P[0][None,:],P[1][None,:],P[2][None,:]
    
    a = B[1,0][:,None]+2*B[4,0][:,None]*x+B[7,0][:,None]*y+B[9,0][:,None]*z
    b = B[2,0][:,None]+2*B[5,0][:,None]*y+B[7,0][:,None]*x+B[8,0][:,None]*z
    c = B[3,0][:,None]+2*B[6,0][:,None]*z+B[8,0][:,None]*y+B[9,0][:,None]*x
    d = B[1,1][:,None]+2*B[4,1][:,None]*x+B[7,1][:,None]*y+B[9,1][:,None]*z
    e = B[2,1][:,None]+2*B[5,1][:,None]*y+B[7,1][:,None]*x+B[8,1][:,None]*z
    f = B[3,1][:,None]+2*B[6,1][:,None]*z+B[8,1][:,None]*y+B[9,1][:,None]*x
    g = B[1,2][:,None]+2*B[4,2][:,None]*x+B[7,2][:,None]*y+B[9,2][:,None]*z
    h = B[2,2][:,None]+2*B[5,2][:,None]*y+B[7,2][:,None]*x+B[8,2][:,None]*z
    i = B[3,2][:,None]+2*B[6,2][:,None]*z+B[8,2][:,None]*y+B[9,2][:,None]*x
    
    det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
    return det


# %%
from scipy.spatial.transform import Rotation as R
from scipy.io import loadmat

import pyro.distributions as dist
import pyro

def simulate_worm_pc(atlas_file,info_file,n_sample):
    '''Generate simulated samples of worm point clouds using an atlas file
    '''
    def sample(theta_l,theta_c,sigma_l,n_sample=n_sample):
        with pyro.plate('neurons',theta_c.size(0)):
            with pyro.plate('data',n_sample):
                sample_c = pyro.sample('obs_c',dist.Dirichlet(theta_c))
                sample_l = pyro.sample('obs_l',dist.MultivariateNormal(theta_l,sigma_l*torch.eye(3)))
        return torch.cat((sample_l,sample_c),2).permute(0,2,1)
    
    content = loadmat(atlas_file,simplify_cells=True)
    neurons = list(content['atlas']['tail']['N'])
    mu = content['atlas']['tail']['model']['mu']
    # sigma = content['atlas']['tail']['model']['sigma']

    data = loadmat(info_file,simplify_cells=True)
    neurons_body = list(set([neuron for ganglion in data['ganglia'][13:21] for neuron in ganglion['neurons']]))
    neurons_body = neurons_body + ['PHSO1L', 'PHSO1R', 'AS10', 'DA7', 'VD11', 'VA11', 'PVPL', 'PVPR']

    ganglia_body = [list(set([neuron for neuron in ganglion['neurons']])) for ganglion in data['ganglia'][13:]]
    ganglia_body += [['AS10', 'DA7', 'VD11','VA11']]
    ganglia_body += [['PHSO1R','PHSO1L']]
        
    tess = [[neurons.index(n) for n in ganglion if n in neurons] for ganglion in ganglia_body]
    tess = [tess[4]+tess[0],tess[0]+tess[1],tess[1]+tess[2]+tess[3]+tess[5]]

    theta_l=torch.tensor(mu[:,:3]).float()
    theta_c=torch.tensor(sp.special.softmax(mu[:,3:]/50,axis=1)).float()
    sigma_l=torch.ones((1)).float()
    point_cloud = sample(theta_l,theta_c,sigma_l)
    for i in range(point_cloud.shape[0]):
        rot = torch.tensor(R.from_rotvec(np.array([0,0,np.random.randn()/3])).as_matrix()).float()
        point_cloud[i,:3,:] = rot@point_cloud[i,:3,:]
    
    return point_cloud, tess


# %%
from dipy.align.transforms import AffineTransform2D, RigidTransform2D, AffineTransform3D, RigidTransform3D
from dipy.align.imaffine import AffineRegistration
import ray


@ray.remote
def affine_register(fixed,moving,transform='rigid'):
    affreg = AffineRegistration()
    if transform == 'rigid':
        if fixed.shape[2] == 1:
            transform = RigidTransform2D()
            fixed = fixed[:,:,0]
            moving = moving[:,:,0]
        else: 
            transform = RigidTransform3D()
    if transform == 'affine':
        if fixed.shape[2] == 1: 
            transform = AffineTransform2D()
            fixed = fixed[:,:,0]
            moving = moving[:,:,0]
        else: 
            transform = AffineTransform3D()
    
    affine = affreg.optimize(
        fixed, moving, 
        transform, params0=None
    )

    registered = affine.transform(moving)
    
    return registered

def affine_atlas(A,dataloader,transform='rigid'):
    refs = []
    for batch_idx, data in enumerate(dataloader):
        refs += [affine_register.remote(
            A[0].numpy(),
            data[0][0,0].numpy(),
            transform
        )]
    registered = ray.get(refs)
    
    return np.array(registered)
    
    
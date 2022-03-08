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
